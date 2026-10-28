from pathlib import Path
import csv
import json
import time
import hashlib
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from statistics import median
import torch
import soundfile as sf
from lrac.train import load_run
from lrac.data.audit import audit
from lrac.data.datasets import AudioDataset
from lrac.config import load_config
from lrac.utils import write_json, get_device, file_hash
from lrac.metrics import waveform_metrics, summarize, speech_quality_metrics


def evaluate(run, split=None, limit=None, device="auto", data_config=None, budgets=None, nmse_targets=None, checkpoint="best.pt", keep_fractions=None, metric_workers=0, resume=False):
    if metric_workers < 0:
        raise ValueError("metric_workers must be nonnegative")
    # Processes avoid the GIL in optional native speech metrics. Bound queued
    # waveforms below so the full corpus is never buffered in memory.
    with ProcessPoolExecutor(max_workers=metric_workers) if metric_workers else nullcontext() as pool:
        return _evaluate(run, split, limit, device, data_config, budgets, nmse_targets, checkpoint, keep_fractions, pool, resume, metric_workers)


def _evaluate(run, split, limit, device, data_config, budgets, nmse_targets, checkpoint, keep_fractions, speech_pool, resume, metric_workers):
    cfg, model, state = load_run(run, device, checkpoint)
    block_model = getattr(model, "uses_procrustes", False)
    sparse_model = getattr(model, "uses_global_budget", False)
    if keep_fractions is not None and (not sparse_model or budgets is not None or nmse_targets is not None):
        raise ValueError("keep_fractions require sparse_autoencoder and cannot be combined with budgets or NMSE targets")
    if keep_fractions is not None and (not keep_fractions or any(not 0 <= value <= 1 for value in keep_fractions)):
        raise ValueError("keep_fractions must be between zero and one")
    if budgets is not None and (not budgets or any(not isinstance(value, int) or value < 0 for value in budgets)):
        raise ValueError("Budgets must be nonnegative integers")
    if nmse_targets is not None and (not block_model or budgets is not None):
        raise ValueError("NMSE targets require block_orthogonal and cannot be combined with budgets")
    if nmse_targets is not None and any(v < 0 for v in nmse_targets):
        raise ValueError("NMSE targets must be nonnegative")
    if data_config:
        external = load_config(data_config)
        if external.data.sample_rate != cfg.data.sample_rate:
            raise ValueError("External evaluation must use the model's sample rate")
        cfg.data = external.data
        cfg.project_root = external.project_root
    output_root = Path(run) / "evaluation"
    if checkpoint != "best.pt":
        output_root = output_root / Path(checkpoint).stem
    audit(cfg, output_root / "audit.json")
    splits = {"validation": cfg.data.val_manifest, **cfg.data.test_manifests}
    if cfg.data.source == "vctk" and cfg.data.vctk_mode == "external":
        splits = {"external": "external.jsonl"}
    selected = list(cfg.data.test_manifests) if split is None else [split]
    if cfg.data.source == "vctk" and cfg.data.vctk_mode == "external" and split is None:
        selected = ["external"]
    results = []
    stage = "smoke" if cfg.train.name.startswith("smoke_") else cfg.train.stage
    target_device = next(model.parameters(), next(model.buffers(), torch.empty(0))).device
    for label in selected:
        if label not in splits:
            raise ValueError(f"Unknown split {label}; choose from {list(splits)}")
        dataset = AudioDataset(cfg.manifest(splits[label]), cfg.data.sample_rate, limit=limit)
        coverage_hash = hashlib.sha256(json.dumps(dataset.rows, sort_keys=True).encode("utf-8")).hexdigest()
        searched = []
        if keep_fractions is not None:
            settings = [("fraction", value) for value in keep_fractions]
        elif nmse_targets is not None:
            settings = [("nmse", value) for value in nmse_targets]
        elif budgets is not None:
            settings = [("budget", value) for value in budgets]
        elif sparse_model:
            settings = [("fraction", value) for value in cfg.evaluation.keep_fractions] if cfg.evaluation.keep_fractions else [("budget", cfg.model.sample_budget)] if cfg.model.sample_budget is not None else [("fraction", cfg.model.keep_fraction)]
        elif block_model:
            if cfg.model.sample_budget is not None:
                settings = [("budget", cfg.model.sample_budget)]
            elif cfg.model.target_nmse is not None:
                settings = [("nmse", cfg.model.target_nmse)]
            else:
                settings = [("fraction", cfg.model.keep_fraction)]
        else:
            settings = [("budget", cfg.model.code_budget)]
        for mode, value in settings:
            code_budget = value if mode == "budget" else None
            if sparse_model:
                model.config.sample_budget = value if mode == "budget" else None
                if mode == "fraction":
                    model.config.keep_fraction = value
            elif block_model:
                model.config.sample_budget = value if mode == "budget" else None
                model.config.target_nmse = value if mode == "nmse" else None
                if mode == "fraction":
                    model.config.keep_fraction = value
            else:
                model.config.code_budget = value
            # Limited runs are deliberately segregated from complete benchmark results.
            out = output_root / label / (f"limit_{limit}" if limit else "smoke" if stage == "smoke" else "complete") / f"{mode}_{value}"
            if resume and (out / "summary.json").exists() and (out / "metrics.csv").exists():
                saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
                if saved["coverage_sha256"] != coverage_hash or saved["checkpoint_step"] != state["step"] or saved["checkpoint"] != checkpoint or saved["representation_setting"] != f"{mode}={value}":
                    raise ValueError("Cannot resume an evaluation with different examples or checkpoint")
                with (out / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                    cached = list(csv.DictReader(stream))
                if len(cached) != len(dataset):
                    raise ValueError("Cached evaluation has incomplete metrics")
                if sparse_model:
                    searched.extend({"id": r["id"], "speaker": r["speaker"],
                        "nmse": float(r["nmse"]) if r["nmse"] else None,
                        "active_coefficients": int(r["active_coefficients"]),
                        "setting": f"{mode}={value}"} for r in cached)
                print(f"REUSED_EVALUATION={out}", flush=True)
                results.append(out)
                continue
            (out / "audio").mkdir(parents=True, exist_ok=True)
            rows, worst, pending = [], [], deque()
            began = time.perf_counter()
            progress_file = out.parent / "progress.json"
            def progress(completed, status="evaluating"):
                write_json(progress_file, {"status": status, "split": label,
                    "representation_setting": f"{mode}={value}",
                    "setting_index": settings.index((mode, value)) + 1,
                    "total_settings": len(settings), "examples_completed": completed,
                    "total_examples": len(dataset), "setting_seconds": time.perf_counter() - began})
            progress(0)
            print(f"EVALUATING split={label} setting={mode}={value} examples={len(dataset)}", flush=True)
            orthogonality_error = None
            if block_model:
                basis = model.matrix()
                orthogonality_error = float((basis.T @ basis - torch.eye(basis.shape[0], device=basis.device, dtype=basis.dtype)).norm())
            for index in range(len(dataset)):
                x, metadata = dataset[index]
                x = x[None].to(target_device)
                lengths = torch.tensor([x.shape[-1]], device=target_device)
                with torch.no_grad():
                    if target_device.type == "cuda":
                        torch.cuda.synchronize(target_device)
                    inference_start = time.perf_counter()
                    estimate, representation = model(x, lengths)
                    if target_device.type == "cuda":
                        torch.cuda.synchronize(target_device)
                    inference_seconds = time.perf_counter() - inference_start
                    values = representation.concatenate()
                    metrics = waveform_metrics(x, estimate, cfg.data.sample_rate)
                seconds = x.shape[-1] / cfg.data.sample_rate
                valid = representation.coefficient_mask
                valid_count = int(valid.sum()) if valid is not None else values.numel()
                active = representation.selected_mask
                if sparse_model:
                    active = (values != 0) & valid
                elif active is None:
                    active = values.abs() > 1e-8
                active_count = int(active.sum())
                full_blocks = representation.timing.get("full_blocks", [values.shape[1]])[0]
                learned_active = active[:, :full_blocks]
                group_active = learned_active.reshape(1, full_blocks, cfg.model.groups, values.shape[-1] // cfg.model.groups) if full_blocks else None
                coding = representation.timing.get("sparse_coding", [{}])[0]
                row = {"id": metadata["id"], "speaker": f"{metadata['corpus']}:{metadata['speaker']}" if metadata.get("speaker") else None, "seconds": seconds, **metrics, "active_fraction": active_count / valid_count, "coefficients_per_second": valid_count / seconds, "active_coefficients_per_second": active_count / seconds, "active_coefficients": active_count, "active_atom_types": int(learned_active.any(dim=1).sum()), "active_feature_dictionaries": int(group_active.any(dim=1).any(dim=-1).sum()) if group_active is not None else 0, "tail_coefficients": int(active[:, full_blocks:].sum()) if block_model else 0, "predicted_nmse": coding.get("predicted_nmse"), "one_fewer_nmse": coding.get("one_fewer_nmse"), "target_nmse": coding.get("target_nmse")}
                row.update(inference_seconds=inference_seconds, values_per_input_sample=active_count / x.shape[-1])
                if cfg.evaluation.speech_metrics:
                    if metadata["corpus"] != "synthetic":
                        if speech_pool is None:
                            row.update(speech_quality_metrics(x, estimate, cfg.data.sample_rate))
                        else:
                            future = speech_pool.submit(speech_quality_metrics, x.detach().cpu().numpy(), estimate.detach().cpu().numpy(), cfg.data.sample_rate)
                            pending.append((row, future))
                            if len(pending) >= 2 * metric_workers:
                                ready_row, ready = pending.popleft()
                                ready_row.update(ready.result())
                    else:
                        row.update(pesq_wb=None, stoi=None, visqol=None, pesq_wb_status="not applicable to synthetic signals", stoi_status="not applicable to synthetic signals", visqol_status="not integrated; no score computed")
                row["minimum_support_verified"] = None
                if mode == "nmse" and metrics["nmse"] is not None:
                    row["minimum_support_verified"] = bool(
                        orthogonality_error < 1e-9
                        and abs(metrics["nmse"] - coding["predicted_nmse"]) < 1e-9
                        and metrics["nmse"] <= value + 1e-9
                        and (active_count == 0 or coding["one_fewer_nmse"] > value)
                    )
                rows.append(row)
                stem = metadata["id"].replace(":", "_").replace("/", "_")
                if index < cfg.evaluation.audio_examples:
                    sf.write(out / "audio" / f"{stem}_reference.wav", x[0, 0].cpu().numpy(), cfg.data.sample_rate, subtype="FLOAT")
                    sf.write(out / "audio" / f"{stem}_reconstructed.wav", estimate[0, 0].cpu().numpy(), cfg.data.sample_rate, subtype="FLOAT")
                if row["snr_db"] is not None:
                    worst.append((row["snr_db"], stem, x[0, 0].cpu().numpy(), estimate[0, 0].cpu().numpy()))
                    worst.sort(key=lambda item: item[0])
                    worst = worst[:3]
                if (index + 1) % 100 == 0:
                    progress(index + 1)
                    print(f"progress setting={mode}={value} examples={index + 1}/{len(dataset)}", flush=True)
            for ready_row, ready in pending:
                ready_row.update(ready.result())
            with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for _, stem, x, y in worst:
                sf.write(out / "audio" / f"worst_{stem}_reference.wav", x, cfg.data.sample_rate, subtype="FLOAT")
                sf.write(out / "audio" / f"worst_{stem}_reconstructed.wav", y, cfg.data.sample_rate, subtype="FLOAT")
            summary = summarize(rows, cfg.evaluation.bootstrap_samples, cfg.train.seed)
            snr = summary["metrics"].get("snr_db", {})
            summary.update({"split": label, "limited": bool(limit), "study_stage": stage, "code_budget": code_budget, "representation_setting": f"{mode}={value}", "budget_scope": "whole recording" if block_model or sparse_model else "per time step", "target_nmse": value if mode == "nmse" else None, "checkpoint_step": state["step"], "checkpoint": checkpoint, "seconds": time.perf_counter() - began, "near_accurate_target_met": snr.get("median", -float("inf")) >= cfg.evaluation.snr_target_median and snr.get("p05", -float("inf")) >= cfg.evaluation.snr_target_p05})
            if sparse_model:
                summary.update(parameters=sum(p.numel() for p in model.parameters()), optimality_scope="achieved support with a learned encoder and magnitude selection; no global minimum certificate", coefficient_precision="float32; coordinate IDs and length metadata additionally required")
            if cfg.evaluation.speech_metrics:
                summary["speech_metric_statuses"] = {key: dict(Counter(row[key] for row in rows)) for key in ["pesq_wb_status", "stoi_status", "visqol_status"]}
            if block_model:
                basis = model.matrix()
                summary["dictionary_values"] = basis.numel()
                summary["orthogonality_error"] = orthogonality_error
                summary["optimality_scope"] = "fixed orthonormal block transform and waveform squared error; not global dictionary or perceptual optimality"
                summary["minimum_support_verified_examples"] = sum(r["minimum_support_verified"] is True for r in rows)
            summary.update({"coverage_sha256": coverage_hash, "manifest_sha256": file_hash(cfg.manifest(splits[label])), "sample_rate": cfg.data.sample_rate, "limit": limit})
            write_json(out / "summary.json", summary)
            if sparse_model:
                searched.extend({**row, "setting": f"{mode}={value}"} for row in rows)
            print(f"EVALUATION={out} median_snr={snr.get('median')}", flush=True)
            progress(len(dataset), "setting complete")
            results.append(out)
        if sparse_model:
            search_rows, target_summaries = [], []
            by_id = {}
            for row in searched:
                by_id.setdefault(row["id"], []).append(row)
            for target in [0.1, 0.01, 0.001]:
                counts, unmatched, silent = [], 0, 0
                for identifier, candidates in by_id.items():
                    valid = [row for row in candidates if row["nmse"] is not None]
                    successful = [row for row in valid if row["nmse"] <= target]
                    best = min(successful, key=lambda row: (row["active_coefficients"], row["nmse"])) if successful else None
                    if best is not None:
                        counts.append(best["active_coefficients"])
                    elif valid:
                        unmatched += 1
                    else:
                        silent += 1
                    search_rows.append({"id": identifier, "speaker": candidates[0]["speaker"], "target_nmse": target, "met": best is not None, "active_coefficients": best["active_coefficients"] if best else None, "achieved_nmse": best["nmse"] if best else None, "setting": best["setting"] if best else None})
                target_summaries.append({"target_nmse": target, "met": len(counts), "unmet": unmatched, "silent": silent, "median_successful_support": median(counts) if counts else None})
            with (out.parent / "support_search.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(search_rows[0]))
                writer.writeheader()
                writer.writerows(search_rows)
            write_json(out.parent / "support_search.json", {"scope": "smallest actual support among the evaluated codes; not a certified minimum", "settings": [f"{mode}={value}" for mode, value in settings], "targets": target_summaries, "coverage_sha256": coverage_hash})
        write_json(out.parent / "progress.json", {"status": "complete", "split": label,
            "representation_setting": f"{mode}={value}", "setting_index": len(settings),
            "total_settings": len(settings), "examples_completed": len(dataset),
            "total_examples": len(dataset)})
    return results
