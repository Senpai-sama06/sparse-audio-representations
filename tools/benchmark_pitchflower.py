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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from lrac.config import load_study_config
from lrac.data.pitchflower import PitchDataset
from lrac.models.pitchflower import PitchFlowerModel, configure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--kind", choices=["rvq", "gs"], default="rvq")
    parser.add_argument("--reference-attention", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", default="runs/pitchflower_benchmark.json")
    parser.add_argument("--apply", action="store_true",
                        help="Migrate the stopped campaign to the fastest tested batch below 13 GiB")
    parser.add_argument("--priority", action="store_true", help="Use production above-normal Windows CPU priority")
    parser.add_argument("--loader", action="store_true", help="Include the real DataLoader and CPU-to-GPU transfer")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.apply and (args.reference_attention or args.kind != "rvq"):
        parser.error("--apply requires optimized RVQ measurements")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    if args.priority:
        configure_runtime()
    settings = load_study_config(ROOT/"configs/study_pitchflower_librispeech.yaml").parameters
    count = 32*(args.steps+2) if args.loader else 32
    rows = [json.loads(line) for line in (ROOT/settings["train_manifest"]).read_text().splitlines()[:count]]
    dataset = PitchDataset(rows, ROOT/settings["cache_dir"], settings["samples"], training=False)
    started = time.perf_counter()
    examples = [dataset[i] for i in range(32)]
    loading_seconds = time.perf_counter()-started
    batch = tuple(torch.stack([example[i] for example in examples]).cuda() for i in range(3))
    report = {"kind": args.kind, "attention": "reference" if args.reference_attention else "sdpa", "priority": args.priority,
              "optimizer_fused": False,
              "includes_loader": args.loader,
              "crop_loading_seconds_32": loading_seconds, "effective_batch": 32, "measurements": []}
    destination = ROOT/args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT/"runs/pitchflower_librispeech/rvq/checkpoints/last.ckpt"
    for size in args.batches:
        if size < 1 or 32 % size:
            raise ValueError("Microbatch must divide effective batch 32")
        model = optimizer = loss = None
        try:
            torch.manual_seed(0)
            model = PitchFlowerModel(args.kind, optimized_attention=not args.reference_attention).cuda().train()
            if args.kind == "rvq":
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model.load_state_dict({k.removeprefix("net."): v for k, v in state["state_dict"].items()})
                del state
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)
            loader = None
            if args.loader:
                loader = torch.utils.data.DataLoader(dataset, batch_size=size,
                    num_workers=settings["loader_workers"], pin_memory=True)
                batches = iter(loader)
            def step():
                optimizer.zero_grad(set_to_none=True)
                for first in range(0, 32, size):
                    microbatch = (tuple(value.cuda(non_blocking=True) for value in next(batches))
                                  if args.loader else tuple(value[first:first+size] for value in batch))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = model(*microbatch)
                    (loss/(32//size)).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                return float(loss.detach()), float(norm)
            # Warm up shape-specific cuDNN plans and optimizer state allocation.
            step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(args.steps):
                start = time.perf_counter()
                value, norm = step()
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-start)
            measurement = dict(micro_batch=size, seconds_per_step=sum(durations)/len(durations),
                durations=durations, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                loss=value, gradient_norm=norm)
        except torch.cuda.OutOfMemoryError:
            measurement = dict(micro_batch=size, status="out_of_memory")
        finally:
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
        config_path = ROOT/"configs/study_pitchflower_librispeech.yaml"
        text = config_path.read_text()
        for key in ("micro_batch", "accumulation", "predictor_accumulation"):
            text = re.sub(rf"(?m)^(  {key}: )\d+", rf"\g<1>{settings[key]}", text)
        config_path.write_text(text)
        print(f"Prepared microbatch {size}; checkpoint step {step}. Training remains stopped.", flush=True)


if __name__ == "__main__":
    main()
