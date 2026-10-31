"""Foreground PitchFlower campaign: prepare, RVQ, GS, predictor, calibration.

Nothing runs on import. The user explicitly launches the campaign through run.py.
Lightning handles AMP/accumulation/checkpoints; the existing sparse-autoencoder
training routine has a different objective and is intentionally not repurposed.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import gc
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from lrac.data.pitchflower import PitchDataset, cached_utterance, cache_lock
from lrac.models.pitchflower import PitchFlowerModel, GaussianPredictor, render_gaussians
from lrac.utils import file_hash, source_fingerprint, environment, write_json


ROOT = Path(__file__).resolve().parents[1]


def _path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT/path


def _records(path):
    return [json.loads(line) for line in _path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_data(settings, output):
    """Require complete, native training utterances and disjoint speaker roles."""
    train = _records(settings["train_manifest"])
    validation = _records(settings["validation_manifest"])
    for rows, split, role in ((train, "train-clean-360", "train"), (validation, "dev-clean", "validation")):
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Empty or duplicate source IDs")
        if any(r["split"] != split or r["role"] != role or r["sample_rate"] != 16000 for r in rows):
            raise ValueError("Unexpected LibriSpeech split, role or sampling rate")
        if any(r.get("start", 0) != 0 or r.get("stop", r["num_samples"]) != r["num_samples"] for r in rows):
            raise ValueError("Use complete source recordings, not an earlier crop/subset manifest")
        if any(not Path(r["path"]).is_file() for r in rows):
            raise FileNotFoundError("Missing source recordings; check cached LibriSpeech paths")
    train_speakers, val_speakers = ({str(r["speaker"]) for r in rows} for rows in (train, validation))
    if train_speakers & val_speakers:
        raise ValueError("Training and validation speakers overlap")
    if len(train) != settings["expected_train_files"] or len(train_speakers) != settings["expected_train_speakers"]:
        raise ValueError("The configured training manifest is not the expected full cached split")
    groups = {}
    for row in sorted(validation, key=lambda r: r["id"]):
        groups.setdefault(str(row["speaker"]), []).append(row)
    # Round-robin sources provide balanced routine validation. Calibration uses
    # different utterances; neither set contains training speakers or test-clean.
    ordered = [group[i] for i in range(max(map(len, groups.values())))
               for group in groups.values() if i < len(group)]
    if len(ordered) < settings["validation_examples"]+40:
        raise ValueError("Not enough disjoint development utterances for validation and calibration")
    routine = ordered[:settings["validation_examples"]]
    calibrated = ordered[settings["validation_examples"]:settings["validation_examples"]+40]
    report = dict(train_files=len(train), train_speakers=len(train_speakers),
        validation_files=len(validation), validation_speakers=len(val_speakers),
        train_hours=sum(r["num_samples"]/16000 for r in train)/3600,
        validation_ids=[r["id"] for r in routine], calibration_ids=[r["id"] for r in calibrated],
        train_manifest_sha256=file_hash(_path(settings["train_manifest"])),
        validation_manifest_sha256=file_hash(_path(settings["validation_manifest"])))
    audit_path = output/"data_audit.json"
    if audit_path.exists() and json.loads(audit_path.read_text()) != report:
        raise ValueError("Dataset changed since campaign creation; use a new run directory")
    write_json(audit_path, report)
    print(f"Data: {len(train):,} training utterances / {len(train_speakers)} speakers / {report['train_hours']:.1f} h", flush=True)
    return train, validation, routine, calibrated


def _prepare_one(task):
    row, cache, seed = task
    torch.set_num_threads(1)
    return str(cached_utterance(row, cache, seed))


def prepare(rows, settings, output):
    """Complete the shared offline cache, with bounded work and resumable files."""
    cache = str(_path(settings["cache_dir"]))
    count = 0
    tasks = iter((row, cache, settings["seed"]) for row in rows)
    with ProcessPoolExecutor(max_workers=settings["prepare_workers"]) as pool:
        pending = deque()
        exhausted = False
        with tqdm(total=len(rows), desc="WORLD preprocessing (cached files reused)", unit="utterance") as bar:
            while pending or not exhausted:
                while len(pending) < 2*settings["prepare_workers"] and not exhausted:
                    task = next(tasks, None)
                    if task is None:
                        exhausted = True
                    else:
                        pending.append(pool.submit(_prepare_one, task))
                if pending:
                    pending.popleft().result()  # Raise errors; never silently omit a file.
                    count += 1
                    bar.update(1)
                    if count % 100 == 0 or count == len(rows):
                        write_json(output/"progress.json", dict(stage="prepare", completed=count, total=len(rows)))


class TrainingModule(pl.LightningModule):
    """One shared CFM training recipe; only the bottleneck-specific term changes."""
    def __init__(self, settings, kind, predictor=False):
        super().__init__()
        self.settings, self.kind, self.predictor_stage = settings, kind, predictor
        self.net = PitchFlowerModel(kind, gs_density=settings["gs_density"], gs_iterations=settings["gs_iterations"],
                                    gs_channels=settings["gs_channels"])
        frames = (settings["samples"]//256+1)//2
        self.predictor = GaussianPredictor(channels=settings["gs_channels"], frames=frames,
            count=max(2, round(frames*settings["gs_density"]))) if predictor else None
        self.last_metrics = {}
        if predictor:
            self.net.requires_grad_(False)
        self.save_hyperparameters()

    def _loss(self, batch, training):
        original, perturbed, pitch = batch
        if self.predictor_stage:
            self.net.eval()
            with torch.no_grad():
                latent = self.net.latents(perturbed)
            count = max(2, round(latent.shape[-1]*self.settings["gs_density"]))
            weights, scales = self.predictor(latent, count)
            loss = F.mse_loss(render_gaussians(weights, scales, latent.shape[-1]).float(), latent.float())
            return loss, {"latent_mse": loss.detach()}
        weight = self.settings["rvq_commitment"] if self.kind == "rvq" else self.settings["gs_commitment"]
        return self.net(original, perturbed, pitch, .1 if training else 0., commitment_weight=weight)

    def training_step(self, batch, batch_idx):
        loss, metrics = self._loss(batch, True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite PitchFlower loss; inspect the input/cache and bottleneck fit")
        self.last_metrics = {key: float(value) for key, value in metrics.items()}
        self.last_metrics["loss"] = float(loss.detach())
        self.log("train/loss", loss, prog_bar=True, batch_size=len(batch[0]))
        for key, value in metrics.items():
            self.log(f"train/{key}", value, batch_size=len(batch[0]))
        return loss

    def validation_step(self, batch, batch_idx):
        # Fixed seeded CFM noise makes comparisons across validations reproducible.
        # fork_rng restores training RNG afterward, including when validation runs
        # in the middle of an epoch.
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type == "cuda" else []):
            torch.manual_seed(self.settings["seed"]+batch_idx)
            loss, _ = self._loss(batch, False)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=len(batch[0]))

    def configure_optimizers(self):
        if not self.predictor_stage:
            return torch.optim.AdamW(self.net.parameters(), lr=self.settings["learning_rate"])
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=5e-4)
        total = self.settings["predictor_steps"]
        def multiplier(step):
            if step < 1000:
                return max(1, step)/1000
            return .1+.9*.5*(1+math.cos(math.pi*min(1., (step-1000)/max(1, total-1000))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class LiveProgress(TQDMProgressBar):
    """Terminal global-step counters plus a small machine-readable progress file."""
    def __init__(self, output, stage, total):
        super().__init__(refresh_rate=1)
        self.output, self.stage, self.total = Path(output), stage, total
        self.previous, self.started = -1, time.monotonic()

    def get_metrics(self, trainer, model):
        metrics = super().get_metrics(trainer, model)
        metrics.update(global_step=trainer.global_step, target_steps=self.total)
        return metrics

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, module, outputs, batch, batch_idx)
        if trainer.global_step != self.previous:
            self.previous = trainer.global_step
            progress = dict(stage=self.stage, status="training",
                step=trainer.global_step, total_steps=self.total, seconds=time.monotonic()-self.started,
                peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30, **module.last_metrics)
            write_json(self.output/"progress.json", progress)
            write_json(self.output.parent/"progress.json", progress)

    def on_exception(self, trainer, module, exception):
        write_json(self.output/"progress.json", dict(stage=self.stage, status="interrupted",
            step=trainer.global_step, total_steps=self.total, error=f"{type(exception).__name__}: {exception}"))


def _loader(rows, settings, training):
    dataset = PitchDataset(rows, _path(settings["cache_dir"]), settings["samples"], settings["seed"], training)
    return DataLoader(dataset, batch_size=settings["micro_batch"] if training else 1,
        shuffle=training, generator=torch.Generator().manual_seed(settings["seed"]),
        num_workers=settings["loader_workers"], pin_memory=True,
        persistent_workers=settings["loader_workers"] > 0)


def _checkpoint(folder):
    """Prefer latest optimizer state; otherwise use the best validated checkpoint."""
    last = folder/"checkpoints/last.ckpt"
    return last if last.exists() else next(iter(sorted((folder/"checkpoints").glob("best*.ckpt"))), None)


def fit(stage, train_rows, val_rows, settings, output):
    """Foreground fit with automatic checkpoint resume; no side effect until called."""
    folder = output/stage
    folder.mkdir(exist_ok=True)
    if (folder/"complete.json").exists():
        print(f"{stage}: already complete", flush=True)
        return
    predictor = stage == "predictor"
    pl.seed_everything(settings["seed"], workers=True)
    module = TrainingModule(settings, "gs" if predictor else stage, predictor)
    last = _checkpoint(folder)
    if predictor and last is None:
        codec = _checkpoint(output/"gs")
        if codec is None:
            raise ValueError("Train the GS codec before its predictor")
        # Only load our locally created checkpoint; optimizer state is not used
        # when moving from codec training to frozen-codec predictor training.
        state = torch.load(codec, map_location="cpu", weights_only=False)["state_dict"]
        module.net.load_state_dict({k.removeprefix("net."): v for k, v in state.items() if k.startswith("net.")})
    total = settings["predictor_steps"] if predictor else settings["steps"]
    checkpoint_dir = folder/"checkpoints"
    best = ModelCheckpoint(dirpath=checkpoint_dir, filename="best_{step}", monitor="val/loss",
                           mode="min", save_top_k=1, save_last=False, save_on_train_epoch_end=False)
    latest = ModelCheckpoint(dirpath=checkpoint_dir, filename="periodic_{step}", save_top_k=0,
        save_last=True, every_n_train_steps=settings["checkpoint_interval"], save_on_train_epoch_end=False)
    accumulation = settings["predictor_accumulation"] if predictor else settings["accumulation"]
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed",
        max_steps=total, max_epochs=-1, accumulate_grad_batches=accumulation,
        gradient_clip_val=1., val_check_interval=settings["validation_interval"]*accumulation,
        check_val_every_n_epoch=None, num_sanity_val_steps=0,
        callbacks=[LiveProgress(folder, stage, total), best, latest],
        logger=TensorBoardLogger(save_dir=folder/"logs", name=""), log_every_n_steps=10,
        default_root_dir=folder, enable_model_summary=True, inference_mode=False)
    print(f"\n{stage}: {sum(p.numel() for p in module.parameters()):,} parameters; "
          f"microbatch {settings['micro_batch']} x accumulation {accumulation}; target {total:,} steps", flush=True)
    try:
        trainer.fit(module, _loader(train_rows, settings, True), _loader(val_rows, settings, False),
                    ckpt_path=str(last) if last else None)
    except KeyboardInterrupt:
        trainer.save_checkpoint(checkpoint_dir/"last.ckpt")
        raise
    if trainer.interrupted:
        trainer.save_checkpoint(checkpoint_dir/"last.ckpt")
        raise KeyboardInterrupt("Campaign interrupted; rerun the same command to resume")
    trainer.save_checkpoint(checkpoint_dir/"last.ckpt")
    export_audio(module, val_rows[:2], settings, folder)
    write_json(folder/"complete.json", dict(step=trainer.global_step, best_checkpoint=best.best_model_path))
    write_json(folder/"progress.json", dict(stage=stage, status="complete", step=trainer.global_step, total_steps=total))
    del trainer, module
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def export_audio(module, rows, settings, folder):
    """Save original and 0/+6/-6-semitone samples using the paper's fixed Vocos."""
    from vocos import Vocos
    import soundfile as sf
    module.eval()
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(module.device).eval()
    destination = folder/"audio"
    destination.mkdir(exist_ok=True)
    dataset = PitchDataset(rows, _path(settings["cache_dir"]), settings["samples"], settings["seed"], False)
    for index in range(len(dataset)):
        original, perturbed, pitch = dataset[index]
        sf.write(destination/f"{index}_original.wav", original.numpy(), 24000, subtype="FLOAT")
        for semitones in (-6, 0, 6):
            changed_pitch = torch.where(pitch > 0, pitch+semitones/12, pitch).to(module.device)[None]
            with torch.random.fork_rng(devices=[module.device.index]):
                torch.manual_seed(settings["seed"]+index)
                mel = module.net.reconstruct_mel(perturbed.to(module.device)[None], changed_pitch,
                    predictor=module.predictor if module.predictor_stage else None)
            waveform = vocoder.decode(mel).flatten().cpu().numpy()
            sf.write(destination/f"{index}_shift{semitones:+d}.wav", waveform, 24000, subtype="FLOAT")


def calibrate(rows, settings, output):
    """Post-training scalar k-means PTQ, without injecting quantization into training."""
    from sklearn.cluster import MiniBatchKMeans
    predictor_file = _checkpoint(output/"predictor")
    if predictor_file is None:
        raise ValueError("Train the GS predictor before calibration")
    module = TrainingModule(settings, "gs", predictor=True)
    module.load_state_dict(torch.load(predictor_file, map_location="cpu", weights_only=False)["state_dict"])
    module = module.cuda().eval()
    parameters = {"weights": [], "scales": []}
    with torch.no_grad():
        for _, perturbed, _ in tqdm(_loader(rows, settings, False), desc="GS post-training calibration"):
            latent = module.net.latents(perturbed.cuda())
            count = max(2, round(latent.shape[-1]*settings["gs_density"]))
            weights, scales = module.predictor(latent, count)
            parameters["weights"].append(weights.cpu().numpy().reshape(-1))
            parameters["scales"].append(scales.cpu().numpy().reshape(-1))
    centers = {}
    for name, values in parameters.items():
        array = np.concatenate(values).reshape(-1, 1)
        estimator = MiniBatchKMeans(n_clusters=64, random_state=settings["seed"], n_init=3, batch_size=4096)
        centers[name] = np.sort(estimator.fit(array).cluster_centers_.reshape(-1))
    np.savez(output/"gs_ptq_6bit.npz", **centers)
    write_json(output/"calibration.json", dict(bits=6, source_ids=[r["id"] for r in rows],
        positions="fixed uniform grid; no position payload", input_normalization=False,
        scope="Bottleneck parameter centroids only; no measured waveform bitstream rate claimed"))


def campaign(config, phase="all"):
    """One explicit terminal command; phases are sequential to fit the 16-GB GPU."""
    settings = config.parameters
    required = {"run_dir", "train_manifest", "validation_manifest", "cache_dir", "expected_train_files",
        "expected_train_speakers", "seed", "samples", "micro_batch", "accumulation", "steps",
        "predictor_steps", "predictor_accumulation", "learning_rate", "loader_workers", "prepare_workers", "validation_examples",
        "validation_interval", "checkpoint_interval", "gs_density", "gs_iterations", "gs_channels", "rvq_commitment", "gs_commitment"}
    if config.kind != "pitchflower" or set(settings) != required:
        raise ValueError(f"PitchFlower settings mismatch: missing={required-set(settings)}, extra={set(settings)-required}")
    if settings["samples"] % 256 or (settings["samples"]//256+1) % 2:
        raise ValueError("The mel frame count must be even for the reference strided architecture")
    if min(settings[k] for k in ("prepare_workers", "micro_batch", "accumulation", "steps", "predictor_steps",
                                  "predictor_accumulation", "validation_interval", "checkpoint_interval")) < 1:
        raise ValueError("Counts must be positive")
    if settings["loader_workers"] < 0 or settings["samples"] < 1024 or not 0 < settings["gs_density"] <= 1:
        raise ValueError("Invalid worker count, crop length or Gaussian density")
    if min(settings["gs_channels"], settings["gs_iterations"], settings["validation_examples"]) < 1:
        raise ValueError("Channel, inner iteration and validation counts must be positive")
    if phase != "prepare" and not torch.cuda.is_available():
        raise RuntimeError("This recipe requires a CUDA GPU; --phase prepare can run on CPU")
    # Setup downloads source/dependencies only. It never runs Trainer.fit itself.
    from tools.setup_pitchflower import setup
    setup(vocoder=phase in ("all", "rvq", "gs", "predictor"))
    output = _path(settings["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    # One foreground campaign owns the output/checkpoint tree at a time.
    with cache_lock(output/"campaign.lock"):
        saved = output/"settings.json"
        if saved.exists() and json.loads(saved.read_text()) != settings:
            raise ValueError("Settings changed; use a new run_dir instead of silently changing a resumed experiment")
        write_json(saved, settings)
        references = {name: subprocess.check_output(["git", "-C", str(ROOT/"runs/vendor"/name), "rev-parse", "HEAD"], text=True).strip()
                      for name in ("PitchFlower", "gs-codec")}
        recipe = {name: file_hash(ROOT/name) for name in ("lrac/pitchflower.py", "lrac/models/pitchflower.py", "lrac/data/pitchflower.py")}
        if (output/"provenance.json").exists():
            if json.loads((output/"provenance.json").read_text())["recipe_sha256"] != recipe:
                raise ValueError("Training recipe changed since checkpoint creation; preserve its source version to resume")
        else:
            write_json(output/"provenance.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
                references=references, recipe_sha256=recipe, source=source_fingerprint(ROOT), environment=environment(),
                papers={name: file_hash(ROOT/"literature"/name) if (ROOT/"literature"/name).exists() else None
                        for name in ("pitchflower.pdf", "gs_bottleneck.pdf")}))
        try:
            train, validation, routine, calibrated = audit_data(settings, output)
            print(f"Campaign outputs: {output}\nPhases: prepare -> RVQ -> GS -> GS predictor -> PTQ", flush=True)
            if phase in ("all", "prepare"):
                prepare(train+validation, settings, output)
            for stage in ("rvq", "gs", "predictor"):
                if phase in ("all", stage):
                    fit(stage, train, routine, settings, output)
            if phase in ("all", "calibrate"):
                calibrate(calibrated, settings, output)
            write_json(output/"progress.json", dict(stage=phase, status="complete"))
        except BaseException as error:
            progress_file = output/"progress.json"
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
            progress.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error=f"{type(error).__name__}: {error}")
            write_json(progress_file, progress)
            raise
