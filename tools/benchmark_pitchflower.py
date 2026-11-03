"""Disposable CUDA throughput checks; never modify campaign weights/checkpoints.

Use real cached crops, with the L0 search left running. Timed optimizer steps
include accumulation, clipping and AdamW; data-loading time is reported separately.
Resident batches isolate GPU execution, so these are not full campaign ETAs.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import re
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from lrac.config import load_study_config
from lrac.data.pitchflower import PitchDataset
from lrac.models.pitchflower import PitchFlowerModel, configure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study_pitchflower_week.yaml")
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--kind", choices=["rvq", "gs"], default="rvq")
    parser.add_argument("--reference-attention", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", default="runs/pitchflower_benchmark.json")
    parser.add_argument("--apply", action="store_true",
                        help="Migrate the stopped campaign to the fastest tested batch below 13 GiB")
    parser.add_argument("--priority", action="store_true", help="Use production above-normal Windows CPU priority")
    parser.add_argument("--loader", action="store_true", help="Include the real DataLoader and CPU-to-GPU transfer")
    parser.add_argument("--shuffle", action="store_true", help="Sample across the whole corpus, as training does")
    parser.add_argument("--workers", type=int, help="Override loader worker count for this disposable check")
    parser.add_argument("--lightning-amp", action="store_true", help="Disable autocast caching, matching Lightning")
    parser.add_argument("--data-seed", type=int, default=0, help="Corpus sample seed; use fresh cohorts for storage checks")
    parser.add_argument("--batch-sweep", action="store_true", help="One update per candidate batch; compare examples/second instead of a fixed effective batch")
    parser.add_argument("--fresh", action="store_true", help="Use fresh weights, without loading an existing campaign checkpoint")
    parser.add_argument("--cached-only", action="store_true", help="Only read prepared mapped sources; do not run WORLD in benchmark workers")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--compressed", action="store_true", help="Read NPZ instead of mapped NPY to compare cold storage throughput")
    parser.add_argument("--cache-metadata", action="store_true", help="Retain NPY headers and F0 in persistent worker RAM")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.apply and (args.reference_attention or args.kind != "rvq"):
        parser.error("--apply requires optimized RVQ measurements")
    if args.apply and (args.batch_sweep or args.compressed):
        parser.error("Batch/storage comparisons require an explicit new configuration; --apply only migrates fixed-batch execution")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    if args.priority:
        configure_runtime()
    settings = load_study_config(ROOT/args.config).parameters
    if args.workers is not None:
        if args.workers < 0:
            parser.error("--workers must be nonnegative")
        settings["loader_workers"] = args.workers
    largest = max(args.batches) if args.batch_sweep else 32
    count = largest*(args.steps+2) if args.loader else largest
    rows = [json.loads(line) for line in (ROOT/settings["train_manifest"]).read_text().splitlines()]
    if args.shuffle:
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(args.data_seed)).tolist()
        rows = [rows[index] for index in order]
    if args.cached_only:
        cache = ROOT/settings["cache_dir"]
        selected = []
        for row in rows:
            identity = hashlib.sha256(f"{row['id']}:{settings['seed']}:pitchflower24k-v1".encode()).hexdigest()
            if (cache/f"{identity}.npy").is_file():
                selected.append(row)
            if len(selected) >= count:
                break
        rows = selected
    else:
        rows = rows[:count]
    if len(rows) < count:
        raise ValueError(f"Need {count} prepared sources, found {len(rows)}")
    dataset = PitchDataset(rows, ROOT/settings["cache_dir"], settings["samples"], training=args.shuffle,
                           prefer_mapped=not args.compressed, cache_metadata=args.cache_metadata)
    started = time.perf_counter()
    examples = [dataset[i] for i in range(largest)]
    loading_seconds = time.perf_counter()-started
    batch = tuple(torch.stack([example[i] for example in examples]).cuda() for i in range(3))
    report = {"kind": args.kind, "attention": "reference" if args.reference_attention else "sdpa", "priority": args.priority,
              "optimizer_fused": False,
              "shuffle": args.shuffle, "loader_workers": settings["loader_workers"],
              "data_seed": args.data_seed,
              "autocast_cache": not args.lightning_amp,
              "includes_loader": args.loader,
              "crop_loading_seconds": loading_seconds, "effective_batch": "candidate" if args.batch_sweep else 32,
              "fresh_weights": args.fresh, "cached_only": args.cached_only,
              "learning_rate": args.learning_rate, "measurements": []}
    report["storage"] = "compressed_npz" if args.compressed else "mapped_npy"
    report["cache_metadata"] = args.cache_metadata
    destination = ROOT/args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT/settings["run_dir"]/"rvq/checkpoints/last.ckpt"
    for size in args.batches:
        print(f"Benchmark {args.kind}: batch {size}, loader={args.loader}, {args.steps} measured updates", flush=True)
        effective = size if args.batch_sweep else 32
        if size < 1 or effective % size:
            raise ValueError("Microbatch must divide effective batch 32")
        model = optimizer = loss = loader = None
        try:
            torch.manual_seed(0)
            model = PitchFlowerModel(args.kind, optimized_attention=not args.reference_attention).cuda().train()
            if args.kind == "rvq" and not args.fresh:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model.load_state_dict({k.removeprefix("net."): v for k, v in state["state_dict"].items()})
                del state
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=False)
            loader = None
            if args.loader:
                loader = torch.utils.data.DataLoader(dataset, batch_size=size,
                    num_workers=settings["loader_workers"], pin_memory=True)
                batches = iter(loader)
            def step():
                optimizer.zero_grad(set_to_none=True)
                loading = 0.
                for first in range(0, effective, size):
                    if args.loader:
                        fetched = time.perf_counter()
                        cpu_batch = next(batches)
                        loading += time.perf_counter()-fetched
                        microbatch = tuple(value.cuda(non_blocking=True) for value in cpu_batch)
                    else:
                        microbatch = tuple(value[first:first+size] for value in batch)
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=not args.lightning_amp):
                        loss, _ = model(*microbatch)
                    (loss/(effective//size)).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                return float(loss.detach()), float(norm), loading
            # Warm up shape-specific cuDNN plans and optimizer state allocation.
            step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            print(f"Batch {size}: initialization/warm-up complete", flush=True)
            durations = []
            waits = []
            for index in range(args.steps):
                start = time.perf_counter()
                value, norm, loading = step()
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-start)
                waits.append(loading)
                if (index+1) % 10 == 0:
                    print(f"Batch {size}: {index+1}/{args.steps} updates, "
                          f"{effective/(sum(durations)/len(durations)):.1f} examples/s", flush=True)
            measurement = dict(micro_batch=size, seconds_per_step=sum(durations)/len(durations),
                durations=durations, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                effective_batch=effective, examples_per_second=effective/(sum(durations)/len(durations)),
                data_wait_seconds_per_step=sum(waits)/len(waits),
                loss=value, gradient_norm=norm)
        except torch.cuda.OutOfMemoryError:
            measurement = dict(micro_batch=size, status="out_of_memory")
        finally:
            if loader is not None:
                del batches, loader
            del model, optimizer, loss
            gc.collect()
            torch.cuda.empty_cache()
        report["measurements"].append(measurement)
        destination.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(measurement), flush=True)
    if args.apply:
        safe = [row for row in report["measurements"] if row.get("peak_gpu_gib", float("inf")) < 13]
        if not safe:
            raise RuntimeError("No measured batch leaves enough GPU memory headroom")
        size = min(safe, key=lambda row: row["seconds_per_step"])["micro_batch"]
        settings.update(micro_batch=size, accumulation=32//size, predictor_accumulation=16//min(size, 16))
        from lrac.pitchflower import migrate_execution
        step = migrate_execution(settings, ROOT/settings["run_dir"])
        config_path = ROOT/args.config
        text = config_path.read_text()
        for key in ("micro_batch", "accumulation", "predictor_accumulation"):
            pattern = rf"(?m)^(  {key}: )\d+"
            if re.search(pattern, text):
                text = re.sub(pattern, rf"\g<1>{settings[key]}", text)
            else:
                # An inherited configuration may not declare execution keys yet.
                text = text.rstrip()+f"\n  {key}: {settings[key]}\n"
        config_path.write_text(text)
        print(f"Prepared microbatch {size}; checkpoint step {step}. Training remains stopped.", flush=True)


if __name__ == "__main__":
    main()
