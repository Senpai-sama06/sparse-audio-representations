"""Epoch training, complete validation, calibration, provenance, and resumption."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import Stopper, atomic_save
from .data import (SAMPLE_RATE, SpeechDataset, collate, digest, prepare,
                   read_manifests, speaker_limit, write_json)
from .metrics import audio_metrics, speech_dependencies, summarize
from .models import AudioSAE, ModelConfig
from .objectives import objective
from .acquire import ensure_manifests
from .runtime import Progress, atomic_json, file_lock


def source_hashes():
    folder = Path(__file__).resolve().parents[1]
    paths = sorted((folder / "sae").glob("*.py")) + sorted(folder.glob("*.py"))
    hashes = {str(p.relative_to(folder)): digest(p) for p in paths}
    return dict(files=hashes, sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=str(torch.__version__), numpy=np.__version__,
                soundfile=importlib.metadata.version("soundfile"), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def make_loader(rows, crop, batch_size, workers, seed, training=False, generator=None):
    dataset = SpeechDataset(rows, crop, training, seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=workers,
                      collate_fn=collate, generator=generator, persistent_workers=False,
                      pin_memory=torch.cuda.is_available())


def write_history(output, history):
    temporary = output / "history.csv.tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output / "history.csv")


def one_epoch(model, loader, device, config, optimizer=None, progress=None):
    training = optimizer is not None
    model.train(training)
    sums, count, metric_rows = {}, 0, []
    started = time.monotonic()
    active_channels = torch.zeros(model.config.latent_width, dtype=torch.bool, device=device)
    with torch.set_grad_enabled(training):
        for step, (x, lengths, identifiers) in enumerate(loader, 1):
            x, lengths = x.to(device), lengths.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            result = model(x, lengths, update_activity=training)
            losses = objective(model, result, x, lengths, config)
            if not all(bool(torch.isfinite(value)) for value in losses.values()):
                raise FloatingPointError(f"Nonfinite {'training' if training else 'validation'} objective")
            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"], error_if_nonfinite=True)
                optimizer.step()
                model.constrain_weights()
            batch = x.shape[0]
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch
            count += batch
            if progress:
                progress.pulse(batch=step, batches=len(loader), current_loss=sums["loss"] / count,
                               gpu_allocated_mb=round(torch.cuda.memory_allocated(device) / 1024**2, 1) if device.type == "cuda" else None,
                               gpu_peak_mb=round(torch.cuda.max_memory_allocated(device) / 1024**2, 1) if device.type == "cuda" else None)
            active_channels |= (result["z"][result["valid"]] != 0).any(0)
            if not training:
                for index, identifier in enumerate(identifiers):
                    size = int(lengths[index])
                    reference = x[index, 0, :size].detach().cpu().numpy()
                    estimate = result["reconstruction"][index, 0, :size].detach().cpu().numpy()
                    metric_rows.append(dict(id=identifier, samples=size,
                                            **audio_metrics(reference, estimate, config["speech_metrics"])))
            if config["log_every"] and step % config["log_every"] == 0:
                print(f"  {'train' if training else 'validation'} {step}/{len(loader)} "
                      f"loss={sums['loss']/count:.6g} elapsed={time.monotonic()-started:.1f}s", flush=True)
    if not count:
        raise ValueError("Empty training or validation loader")
    means = {name: value / count for name, value in sums.items()}
    means.update(examples=count, inactive_channels=int((~active_channels).sum()),
                 seconds=time.monotonic() - started)
    if not training:
        means.update({"metric_" + name: value for name, value in summarize(metric_rows).items()})
    return means, metric_rows


def run(config, resume=None):
    config = dict(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (Path(resume).resolve().parent if resume else
              Path(config["output"] or f"runs/sparse_ae/{config['method']}/{stamp}_seed{config['seed']}").resolve())
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; use --resume or a new --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config["output"] = str(output)
    with file_lock(output / ".train.lock", timeout=0):
        progress = Progress(output, config["method"])
        try:
            return _run(config, resume, progress)
        except BaseException as error:
            progress.update(phase="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            raise


def _run(config, resume, progress):
    config = dict(config)
    config["channels"] = list(config["channels"])
    if config["speech_metrics"]:
        speech_dependencies()
    torch.set_num_threads(config["threads"])
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device=device)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if config["deterministic"]:
        torch.use_deterministic_algorithms(True)
    checkpoint = None
    if resume:
        # last.pt contains Python/NumPy optimizer and RNG objects; use our own run artifacts.
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        output = Path(resume).resolve().parent
    else:
        output = Path(config["output"])
    manifest_dir, data_source = ensure_manifests(config, progress)
    atomic_json(output / "data_source.json", data_source)
    progress.update(phase="checking_data", manifest_dir=str(manifest_dir))
    partitions, data_hash = read_manifests(manifest_dir, config["verify_audio_hashes"])
    sources = source_hashes()
    if checkpoint:
        if checkpoint["source_sha256"] != sources["sha256"]:
            raise ValueError("Training source changed; resume requires identical source")
        if checkpoint["data_sha256"] != data_hash:
            raise ValueError("Data manifests changed since checkpoint")
    train_rows = speaker_limit(partitions["train"], config["train_limit"])
    val_rows = speaker_limit(partitions["validation"], config["validation_limit"])
    cal_rows = speaker_limit(partitions["calibration"], config["calibration_limit"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = make_loader(train_rows, round(config["crop_seconds"] * SAMPLE_RATE),
                               config["batch_size"], config["workers"], config["seed"], True, generator)
    # Separate generators prevent calibration/validation iteration from changing training shuffle.
    val_loader = make_loader(val_rows, round(config["validation_crop_seconds"] * SAMPLE_RATE),
                             1, config["workers"], config["seed"],
                             generator=torch.Generator().manual_seed(config["seed"] + 1))
    cal_loader = make_loader(cal_rows, round(config["crop_seconds"] * SAMPLE_RATE),
                             config["batch_size"], config["workers"], config["seed"],
                             generator=torch.Generator().manual_seed(config["seed"] + 2))
    model_config = ModelConfig(**{key: config[key] for key in
                                 ("method", "channels", "latent_width", "k", "threshold", "bandwidth",
                                  "dead_after_steps", "aux_k")})
    model = AudioSAE(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    stopper = Stopper(config["patience"], config["min_delta"], config["min_epochs"])
    history, start_epoch = [], 1
    best_payload, best_validation = None, None
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # CPU-loaded optimizer state must follow the parameters to the selected device.
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value) and name != "step":
                    state[name] = value.to(device)
        stopper = Stopper(**checkpoint["stopper"])
        stopper.patience, stopper.min_delta, stopper.min_epochs = (
            config["patience"], config["min_delta"], config["min_epochs"])
        history, start_epoch = checkpoint["history"], checkpoint["epoch"] + 1
        best_payload, best_validation = checkpoint["best_payload"], checkpoint["best_validation"]
        restore_rng(checkpoint["rng"], generator)
    output.mkdir(parents=True, exist_ok=True)
    config.update(output=str(output), manifest_dir=str(manifest_dir))
    write_json(output / "config.json", config)
    if not checkpoint:
        write_json(output / "initial_config.json", config)
        write_json(output / "environment.json", environment())
        write_json(output / "source.json", sources)
        write_json(output / "data_selection.json", dict(
            manifest_sha256=data_hash,
            train_ids=[r["id"] for r in train_rows], validation_ids=[r["id"] for r in val_rows],
            calibration_ids=[r["id"] for r in cal_rows],
            epoch_definition="One deterministic random crop per selected training utterance; shuffled once.",
            validation="Full utterances" if not config["validation_crop_seconds"] else "Fixed centered crops",
            test_used=False))
    elif best_payload is not None:
        # Resuming an older periodic checkpoint restores its own best artifacts.
        atomic_save(best_payload, output / "best_weights.pt")
        atomic_save(dict(format_version=1, model_config=best_payload["model_config"],
                         decoder_state={k.removeprefix("decoder."): v
                                        for k, v in best_payload["model_state"].items() if k.startswith("decoder.")},
                         epoch=best_payload["epoch"], sample_rate=SAMPLE_RATE), output / "decoder_best.pt")
        write_json(output / "best_validation.json", best_validation)
        if history:
            write_history(output, history)
        checkpoint["config"] = config
        atomic_save(checkpoint, output / "last.pt")
    print(f"RUN={output}\nMETHOD={config['method']} DEVICE={device} "
          f"PARAMETERS={sum(p.numel() for p in model.parameters())}\n"
          f"UTTERANCES train={len(train_rows)} validation={len(val_rows)} calibration={len(cal_rows)}\n"
          f"STOP max_epochs={config['epochs']} patience={stopper.patience} "
          f"min_delta={stopper.min_delta} monitor={config['monitor']}", flush=True)
    stop_reason = "max_epochs"
    # An already stopped run is not silently restarted with the same stopping settings.
    already_stopped = (stopper.patience > 0 and stopper.bad_epochs >= stopper.patience and
                       start_epoch - 1 >= stopper.min_epochs)
    if already_stopped:
        stop_reason = "early_stopping"
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            if already_stopped:
                break
            train_loader.dataset.epoch = epoch
            progress.update(phase="training", epoch=epoch, max_epochs=config["epochs"],
                            completed_epochs=epoch - 1, batch=0, batches=len(train_loader),
                            current_loss=None,
                            best_epoch=stopper.best_epoch,
                            best_value=stopper.best if math.isfinite(stopper.best) else None)
            print(f"Epoch {epoch}/{config['epochs']}", flush=True)
            train, _ = one_epoch(model, train_loader, device, config, optimizer, progress)
            progress.update(phase="calibrating", batch=0, batches=len(cal_loader))
            model.calibrate(cal_loader, device, progress)
            progress.update(phase="validating", batch=0, batches=len(val_loader), current_loss=None)
            validation, metrics = one_epoch(model, val_loader, device, config, progress=progress)
            value = validation[config["monitor"]]
            improved, should_stop = stopper.update(value, epoch)
            row = dict(epoch=epoch, **{"train_" + k: v for k, v in train.items()},
                       **{"val_" + k: v for k, v in validation.items()},
                       inference_threshold=float(model.inference_threshold),
                       bad_epochs=stopper.bad_epochs, best_epoch=stopper.best_epoch)
            history.append(row)
            progress.update(phase="checkpointing", validation_value=value,
                            best_epoch=stopper.best_epoch, best_value=stopper.best,
                            bad_epochs=stopper.bad_epochs, patience=stopper.patience)
            weights = dict(format_version=1, model_config=model.specification(),
                           model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                           epoch=epoch, monitor=config["monitor"], monitor_value=value,
                           data_sha256=data_hash, source_sha256=sources["sha256"],
                           stage="native_objective_unquantized", sample_rate=SAMPLE_RATE)
            if improved:
                best_payload = weights
                best_validation = dict(epoch=epoch, metrics=metrics, summary=validation)
                atomic_save(weights, output / "best_weights.pt")
                atomic_save(dict(format_version=1, model_config=model.specification(),
                                 decoder_state={k: v.detach().cpu() for k, v in model.decoder.state_dict().items()},
                                 epoch=epoch, sample_rate=SAMPLE_RATE), output / "decoder_best.pt")
                write_json(output / "best_validation.json", best_validation)
            state = dict(**weights, config=config, optimizer_state=optimizer.state_dict(),
                         stopper=stopper.state(), history=history, rng=rng_state(generator),
                         best_payload=best_payload, best_validation=best_validation)
            atomic_save(state, output / "last.pt")
            if config["save_every"] and epoch % config["save_every"] == 0:
                atomic_save(state, output / f"epoch_{epoch:04d}.pt")
            write_history(output, history)
            progress.update(completed_epochs=epoch, last_checkpoint=str(output / "last.pt"))
            print(f"  val_{config['monitor']}={value:.7g} best={stopper.best:.7g} "
                  f"best_epoch={stopper.best_epoch} stale={stopper.bad_epochs}/{stopper.patience} "
                  f"active/step={validation['active_per_step']:.2f}", flush=True)
            if should_stop:
                stop_reason = "early_stopping"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("Interrupted. last.pt retains the last completed epoch; resume repeats any incomplete epoch.", flush=True)
    completed = history[-1]["epoch"] if history else 0
    summary = dict(method=config["method"], completed_epochs=completed, stop_reason=stop_reason,
                   best_epoch=stopper.best_epoch, best_value=stopper.best if math.isfinite(stopper.best) else None,
                   monitor=config["monitor"], best_weights=str(output / "best_weights.pt") if stopper.best_epoch else None,
                   resume_checkpoint=str(output / "last.pt") if completed else None,
                   stage="native_objective_unquantized", test_used=False)
    write_json(output / "summary.json", summary)
    progress.update(phase="interrupted" if stop_reason == "interrupted" else "completed",
                    epoch=completed, max_epochs=config["epochs"], completed_epochs=completed, best_epoch=stopper.best_epoch,
                    best_value=summary["best_value"], stop_reason=stop_reason,
                    last_checkpoint=summary["resume_checkpoint"])
    print(json.dumps(summary, indent=2), flush=True)
    return summary
