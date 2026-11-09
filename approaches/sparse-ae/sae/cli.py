"""The same CLI contract for every architecture-specific entry script."""
import argparse
import math
from pathlib import Path
import sys

from .models import ModelConfig
from .training import run
from .acquire import DEFAULT_DATASET


def parser_for(method):
    parser = argparse.ArgumentParser(
        description=f"Train the {method} audio autoencoder; reuse local LibriSpeech or acquire it through KaggleHub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data = parser.add_argument_group("LibriSpeech data")
    data.add_argument("--root", help="LibriSpeech root or its parent; not needed with --manifest-dir")
    data.add_argument("--manifest-dir", help="Directory made by prepare_data.py; reuse for matched experiments")
    data.add_argument("--dataset", default=DEFAULT_DATASET, help="KaggleHub fallback handle if the corpus is absent")
    data.add_argument("--cache-dir", help="Shared setup/manifest cache; default SPARSE_AE_CACHE or ~/.cache/sparse-ae")
    data.add_argument("--offline", action="store_true", help="Use existing data only; never install or download")
    data.add_argument("--train-split", default="train-clean-100")
    data.add_argument("--dev-split", default="dev-clean")
    data.add_argument("--test-split", default="test-clean", help="Optional manifest only; never scored by training")
    data.add_argument("--train-hours", type=float, help="Speaker-round-robin subset; omit to use full training split")
    data.add_argument("--crop-seconds", type=float, default=2.0, help="One random crop per utterance per training epoch")
    data.add_argument("--validation-crop-seconds", type=float, default=0, help="0 means full utterances; >0 selects centered crops")
    data.add_argument("--train-limit", type=int, default=0, help="0 means all selected training utterances")
    data.add_argument("--validation-limit", type=int, default=256, help="Speaker-round-robin selection; 0 means all")
    data.add_argument("--calibration-limit", type=int, default=256, help="BatchTopK threshold calibration; 0 means all")
    data.add_argument("--verify-audio-hashes", action="store_true", help="Rehash audio in addition to checking sizes/timestamps")
    model = parser.add_argument_group("Shared backbone and method settings")
    model.add_argument("--channels", type=int, nargs=4, default=[32, 64, 128, 128])
    model.add_argument("--latent-width", type=int, default=256)
    model.add_argument("--k", type=int, default=16, help="Top-K/BatchTopK channels per latent time step")
    model.add_argument("--threshold", type=float, default=.1, help="Initial positive JumpReLU threshold")
    model.add_argument("--bandwidth", type=float, default=.1, help="JumpReLU rectangular gradient-kernel bandwidth")
    model.add_argument("--sparsity-weight", type=float,
                       default={"l1": 1e-4, "kl": 1e-3, "jumprelu": 1e-3, "gated": 1e-4}.get(method, 0.0))
    model.add_argument("--kl-target", type=float, default=.05)
    model.add_argument("--gated-aux-weight", type=float, default=1.0)
    model.add_argument("--aux-weight", type=float, default=.03, help="Top-K/BatchTopK dead-latent residual objective")
    model.add_argument("--aux-k", type=int, default=32)
    model.add_argument("--dead-after-steps", type=int, default=100)
    training = parser.add_argument_group("Training, validation, and stopping")
    training.add_argument("--epochs", "--max-epochs", dest="epochs", type=int, default=100)
    training.add_argument("--patience", type=int, default=8, help="Stop after this many stale validation epochs; 0 disables")
    training.add_argument("--min-delta", type=float, default=1e-4, help="Absolute validation improvement required to reset patience")
    training.add_argument("--min-epochs", type=int, default=1, help="Do not stop early before this epoch")
    training.add_argument("--monitor", choices=["loss", "reconstruction"], default="loss",
                          help="Validation objective to minimize; loss includes method-specific penalties")
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--clip-grad", type=float, default=1.0)
    training.add_argument("--mse-scale", type=float, default=.1, help="Fixed waveform amplitude scale; MSE is divided by its square")
    training.add_argument("--spectral-weight", type=float, default=.1)
    training.add_argument("--speech-metrics", action="store_true", help="Also compute validation PESQ-WB and STOI; needs optional packages")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    training.add_argument("--gpu-memory-fraction", type=float, default=1.0,
                          help="Per-process PyTorch CUDA allocator cap (not a total GPU memory reservation)")
    training.add_argument("--workers", type=int, default=0, help="Windows-safe DataLoader worker count")
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--deterministic", action="store_true", help="Require deterministic Torch algorithms; CUDA may need CUBLAS_WORKSPACE_CONFIG")
    training.add_argument("--log-every", type=int, default=50, help="Batch progress interval; 0 disables")
    artifacts = parser.add_argument_group("Artifacts")
    artifacts.add_argument("--output", help="New/empty run directory; default runs/sparse_ae/METHOD/TIMESTAMP_seedN")
    artifacts.add_argument("--resume", help="Path to this method's last.pt or epoch_NNNN.pt")
    artifacts.add_argument("--save-every", type=int, default=0, help="Also keep an epoch checkpoint every N epochs; 0 keeps best and last")
    return parser


def validate(config):
    for key in ("epochs", "min_epochs", "batch_size", "threads", "dead_after_steps", "aux_k"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("patience", "workers", "train_limit", "validation_limit", "calibration_limit", "save_every", "log_every"):
        if config[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("crop_seconds", "learning_rate", "clip_grad", "mse_scale", "threshold", "bandwidth"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if round(config["crop_seconds"] * 16000) < 1:
        raise ValueError("crop_seconds is shorter than one sample")
    for key in ("min_delta", "validation_crop_seconds", "spectral_weight", "sparsity_weight", "aux_weight", "gated_aux_weight"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 < config["kl_target"] < 1:
        raise ValueError("kl_target must lie strictly between 0 and 1")
    if not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if not 0 < config["gpu_memory_fraction"] <= 1:
        raise ValueError("gpu_memory_fraction must be in (0, 1]")
    if config["train_hours"] is not None and (not math.isfinite(config["train_hours"]) or config["train_hours"] <= 0):
        raise ValueError("train_hours must be finite and positive")
    ModelConfig(**{key: config[key] for key in
                   ("method", "channels", "latent_width", "k", "threshold", "bandwidth", "dead_after_steps", "aux_k")}).validate()


def main(method, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = parser_for(method)
    parsed = vars(parser.parse_args(argv))
    resume = parsed.pop("resume")
    explicit = {parser._option_string_actions[token.split("=", 1)[0]].dest
                for token in argv if token.split("=", 1)[0] in parser._option_string_actions}
    for key in ("root", "manifest_dir", "output", "cache_dir"):
        if parsed[key]:
            parsed[key] = str(Path(parsed[key]).resolve())
    if resume:
        import torch
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if not {"config", "optimizer_state", "stopper", "rng"}.issubset(saved):
            parser.error("--resume needs last.pt or epoch_NNNN.pt, not best_weights.pt")
        config = dict(saved["config"])
        if config["method"] != method:
            parser.error(f"Checkpoint method is {config['method']}, entry script is {method}")
        mutable = {"epochs", "patience", "min_delta", "min_epochs", "device", "gpu_memory_fraction", "workers", "threads", "log_every", "save_every", "verify_audio_hashes"}
        for key in explicit - {"resume"}:
            if key not in mutable and parsed[key] != config[key]:
                parser.error(f"Cannot change {key} on resume; use a new run for changed training settings")
            config[key] = parsed[key]
        if Path(resume).resolve().parent != Path(config["output"]):
            parser.error("Resume the checkpoint from its original run directory")
    else:
        config = dict(parsed, method=method)
        if config["root"] and config["manifest_dir"]:
            parser.error("Choose --root to prepare data or --manifest-dir to reuse it")
        if config["manifest_dir"] and config["train_hours"] is not None:
            parser.error("Set --train-hours when preparing manifests; prepared manifests already fix the subset")
    try:
        validate(config)
        return run(config, resume)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
