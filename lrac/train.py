from datetime import datetime, timezone
from pathlib import Path
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lrac.config import from_dict
from lrac.data.audit import audit
from lrac.data.datasets import AudioDataset, collate
from lrac.losses import reconstruction_loss, functional_interaction
from lrac.models import build_model
from lrac.utils import seed_all, get_device, environment, write_json, atomic_checkpoint, source_fingerprint


def _loader(cfg, manifest, training=False, limit=None):
    dataset = AudioDataset(manifest, cfg.data.sample_rate, round(cfg.data.crop_seconds * cfg.data.sample_rate) if training else None, limit)
    generator = torch.Generator().manual_seed(cfg.train.seed)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size if training else 1, sampler=dataset.sampler(generator) if training else None, shuffle=False, num_workers=cfg.train.workers, collate_fn=collate, generator=generator, pin_memory=torch.cuda.is_available())
    return loader


def _infinite(loader):
    while True:
        yield from loader


@torch.no_grad()
def validation(model, loader, device, power, limit, keep_fractions=None, spectral_weight=0.1):
    model.eval()
    scores = []
    per_fraction = {str(value): [] for value in (keep_fractions or [])}
    for i, (x, lengths, _) in enumerate(loader):
        if i >= limit:
            break
        x, lengths = x.to(device), lengths.to(device)
        if keep_fractions:
            dense = model.encode_dense(x, lengths)
            for fraction in keep_fractions:
                estimate = model.decode(model.sparsify(dense, keep_fraction=fraction))
                value = float(reconstruction_loss(x, estimate, lengths, power, spectral_weight=spectral_weight))
                scores.append(value)
                per_fraction[str(fraction)].append(value)
        else:
            estimate, _ = model(x, lengths)
            scores.append(float(reconstruction_loss(x, estimate, lengths, power, spectral=not model.is_dictionary, spectral_weight=spectral_weight)))
    if not scores:
        raise ValueError("Validation has no examples")
    model.validation_scores = {key: sum(values) / len(values) for key, values in per_fraction.items()}
    return sum(scores) / len(scores)


def _rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "python": random.getstate()}


def _gradient_update(model, optimizer, iterator, cfg, power, device, step):
    optimizer.zero_grad(set_to_none=True)
    stats = {"loss": 0.0, "reconstruction": 0.0, "functional": 0.0, "sparsity": 0.0, "group": 0.0, "active_fraction": 0.0, "sampled_keep_fraction": 0.0}
    warmup = round(cfg.train.steps * cfg.train.warmup_fraction)
    ramp = min(1.0, max(0.0, (step - warmup) / max(1, warmup)))
    for micro in range(cfg.train.accumulation):
        x, lengths, _ = next(iterator)
        x, lengths = x.to(device), lengths.to(device)
        if cfg.train.keep_fractions:
            fraction = random.choice(cfg.train.keep_fractions)
            representation = model.sparsify(model.encode_dense(x, lengths), keep_fraction=fraction)
            estimate = model.decode(representation)
            stats["sampled_keep_fraction"] += fraction / cfg.train.accumulation
        else:
            estimate, representation = model(x, lengths)
        rec = reconstruction_loss(x, estimate, lengths, power, spectral=not model.is_dictionary, spectral_weight=cfg.train.spectral_weight)
        values = representation.concatenate()
        valid_time = torch.arange(values.shape[1], device=device)[None, :] * representation.timing["hop"] < lengths[:, None]
        active_values = values[valid_time]
        stats["active_fraction"] += float((active_values != 0).float().mean().detach()) / cfg.train.accumulation
        l1 = active_values.abs().mean()
        grouped = active_values.reshape(-1, cfg.model.groups, values.shape[-1] // cfg.model.groups)
        group = grouped.norm(dim=-1).mean()
        func = rec.new_zeros(())
        if cfg.train.functional_weight and step % cfg.train.functional_interval == 0 and ramp > 0 and micro == 0:
            single = representation.with_values(values[:1])
            single.lengths = lengths[:1]
            width = min(x.shape[-1], round(cfg.train.functional_window_ms * cfg.data.sample_rate / 1000))
            start = int(torch.randint(max(1, int(lengths[0]) - width + 1), (1,)).item())
            def synthesis(z):
                return model.decode(single.with_values(z))[..., start:start + width]
            func = functional_interaction(synthesis, values[:1].float(), cfg.model.groups, cfg.train.functional_probes)
        loss = rec + ramp * (cfg.train.sparsity_weight * l1 + cfg.train.group_weight * group + cfg.train.functional_weight * cfg.train.functional_interval * func)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        (loss / cfg.train.accumulation).backward()
        for key, val in [("loss", loss), ("reconstruction", rec), ("functional", func), ("sparsity", l1), ("group", group)]:
            stats[key] += float(val.detach()) / cfg.train.accumulation
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.train.clip_grad)
    optimizer.step()
    model.post_step()
    return stats


def train(cfg, resume=None):
    seed_all(cfg.train.seed)
    device = get_device(cfg.train.device)
    if resume:
        resume = Path(resume)
        run = resume.parent.parent
        resume_state = torch.load(resume, map_location=device, weights_only=True)
        checkpoint_cfg = from_dict(resume_state["config"]).to_dict()
        if checkpoint_cfg["model"] != cfg.to_dict()["model"]:
            raise ValueError("Resume model configuration differs from checkpoint")
        if checkpoint_cfg["data"] != cfg.to_dict()["data"]:
            raise ValueError("Resume dataset configuration differs from checkpoint")
        if resume_state["config"]["train"]["seed"] != cfg.train.seed:
            raise ValueError("Resume seed differs from checkpoint")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = cfg.path(cfg.train.runs_dir) / f"{cfg.train.name}_{stamp}_seed{cfg.train.seed}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"RUN={run}", flush=True)
    report = audit(cfg, run / "audit.json")
    write_json(run / "config.json", cfg.to_dict())
    write_json(run / "environment.json", environment())
    write_json(run / "source.json", source_fingerprint(cfg.project_root))
    loader = _loader(cfg, cfg.manifest(cfg.data.train_manifest), training=True)
    val_loader = _loader(cfg, cfg.manifest(cfg.data.val_manifest), limit=cfg.train.val_examples)
    model = build_model(cfg.model, cfg.data.sample_rate).to(device)
    samples = 0
    energy = 0.0
    # Deterministic estimation independent of random crop selection in the training loader.
    for x, lengths, _ in _loader(cfg, cfg.manifest(cfg.data.train_manifest), limit=min(64, len(loader.dataset))):
        for item, n in zip(x, lengths):
            energy += float(item[..., :int(n)].double().square().sum())
            samples += int(n)
    power = max(energy / max(samples, 1), 1e-8)
    parameters = [p for p in model.parameters() if p.requires_grad]
    procrustes = getattr(model, "uses_procrustes", False)
    if not parameters:
        optimizer = None
    elif model.is_dictionary:
        optimizer = torch.optim.Adam(parameters, lr=cfg.train.learning_rate)
    else:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    start_step, best = 0, float("inf")
    if resume:
        state = resume_state
        model.load_state_dict(state["model"])
        if optimizer and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        start_step, best, power = state["step"], state["best_validation"], state["train_power"]
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if device.type == "cuda" and state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all([v.cpu() for v in state["rng"]["cuda"]])
        random.setstate(state["rng"]["python"])
    elif model.needs_fit:
        model.fit(loader, device, cfg.train.fit_frames)
    write_json(run / "training_statistics.json", {"train_power": power, "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in parameters), "dictionary_values": model.matrix().numel() if hasattr(model, "matrix") else None, "training_method": "orthogonal_procrustes" if procrustes else "gradient" if optimizer else "fixed_or_fitted", "manifest_audit_passed": report["passed"], "resume_note": "Model/optimizer RNG is restored; random crop/sampler position restarts at resume."})
    writer = SummaryWriter(str(run / "tensorboard"))
    history = run / "history.jsonl"
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def save(step, val, name):
        payload = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None, "config": cfg.to_dict(), "step": step, "best_validation": best, "validation": val, "train_power": power, "rng": _rng_state()}
        atomic_checkpoint(run / "checkpoints" / name, payload)

    def validate(step):
        fractions = cfg.train.validation_keep_fractions or cfg.train.keep_fractions
        value = validation(model, val_loader, device, power, cfg.train.val_examples, fractions, cfg.train.spectral_weight)
        with (run / "validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step, "loss": value, "keep_fraction_losses": model.validation_scores}) + "\n")
        return value

    try:
        score = validate(start_step)
        if score < best:
            best = score
            save(start_step, score, "best.pt")
        if not resume:
            save(start_step, score, "initial.pt")
        save(start_step, score, "last.pt")
        iterator = _infinite(loader)
        last_step = start_step
        for step in range(start_step + 1, cfg.train.steps + 1) if optimizer or procrustes else []:
            model.train()
            if procrustes:
                x, lengths, _ = next(iterator)
                x, lengths = x.to(device), lengths.to(device)
                stats = model.update_dictionary(x, lengths)
                stats.update({"reconstruction": stats["mse_after"] / power, "loss": stats["mse_after"] / power, "functional": 0.0})
            else:
                stats = _gradient_update(model, optimizer, iterator, cfg, power, device, step)
            last_step = step
            if step % cfg.train.log_interval == 0 or step == 1:
                for key, val in stats.items():
                    writer.add_scalar(f"train/{key}", val, step)
                with history.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": step, "seconds": time.perf_counter() - started, **stats}) + "\n")
                write_json(run / "progress.json", {"status": "training", "step": step, "total_steps": cfg.train.steps, "best_validation": best, "seconds": time.perf_counter() - started, **stats})
                print(f"step={step} reconstruction={stats['reconstruction']:.6g} functional={stats['functional']:.6g}", flush=True)
            if step % cfg.train.val_interval == 0 or step == cfg.train.steps:
                score = validate(step)
                writer.add_scalar("validation/reconstruction", score, step)
                print(f"validation step={step} loss={score:.6g}", flush=True)
                if score < best:
                    best = score
                    save(step, score, "best.pt")
                save(step, score, "last.pt")
            elif step % cfg.train.checkpoint_interval == 0:
                save(step, score, "last.pt")
        completion = {"completed": True, "step": last_step, "best_validation": best, "seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
        write_json(run / "completion.json", completion)
        write_json(run / "progress.json", {"status": "complete", "total_steps": cfg.train.steps, **completion})
    except (Exception, KeyboardInterrupt) as error:
        write_json(run / "progress.json", {"status": "failed", "error": str(error), "seconds": time.perf_counter() - started})
        raise
    finally:
        writer.close()
    print(f"RUN={run}", flush=True)
    return run


def load_run(run, device="auto", checkpoint="best.pt"):
    run = Path(run)
    cfg = from_dict(json.loads((run / "config.json").read_text(encoding="utf-8")))
    device = get_device(device)
    state = torch.load(run / "checkpoints" / checkpoint, map_location=device, weights_only=True)
    model = build_model(cfg.model, cfg.data.sample_rate).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return cfg, model, state
