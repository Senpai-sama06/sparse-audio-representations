"""Strict, composable configurations; relative paths resolve against the project."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import copy
import math
import re
import yaml


@dataclass
class DataConfig:
    source: str = "synthetic"
    root: str = "data/raw"
    manifest_dir: str = "data/manifests/synthetic"
    sample_rate: int = 16000
    train_manifest: str = "train.jsonl"
    val_manifest: str = "validation.jsonl"
    test_manifests: dict = field(default_factory=lambda: {"iid": "test_iid.jsonl", "combinations": "test_combinations.jsonl", "extrapolation": "test_extrapolation.jsonl", "families": "test_families.jsonl"})
    train_count: int = 10000
    validation_count: int = 1000
    test_count: int = 1000
    synthetic_seconds: float = 1.0
    subsets_hours: list = field(default_factory=lambda: [1, 10])
    librispeech_train_split: str = "train-clean-100"
    vctk_per_speaker: int = 10
    vctk_mode: str = "external"
    expected_speakers: dict = field(default_factory=dict)
    crop_seconds: float = 2.0
    hash_audio: bool = True


@dataclass
class ModelConfig:
    kind: str = "orthogonal"
    frame_length: int = 512
    hop: int = 128
    atoms: int = 1024
    groups: int = 4
    threshold: float = 0.01
    group_threshold: float = 0.0
    inference_steps: int = 200
    inference_tolerance: float = 1e-5
    code_budget: int = 0
    conv_kernels: int = 64
    conv_length: int = 512
    conv_hop: int = 64
    latent_width: int = 128
    channels: list = field(default_factory=lambda: [32, 64, 128, 128])
    trainable_kernels: bool = True
    initialization: str = "dct"
    keep_fraction: float = 0.125
    sample_budget: int | None = None
    target_nmse: float | None = None
    activation: str = "leaky_relu"


@dataclass
class TrainConfig:
    name: str = "experiment"
    stage: str = "main"
    seed: int = 0
    device: str = "auto"
    steps: int = 20000
    batch_size: int = 8
    accumulation: int = 1
    workers: int = 0
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    clip_grad: float = 1.0
    val_interval: int = 200
    val_examples: int = 64
    log_interval: int = 20
    checkpoint_interval: int = 1000
    fit_frames: int = 250000
    sparsity_weight: float = 0.0
    group_weight: float = 0.0
    functional_weight: float = 0.0
    functional_interval: int = 4
    functional_probes: int = 4
    functional_window_ms: float = 128.0
    warmup_fraction: float = 0.1
    runs_dir: str = "runs"
    keep_fractions: list = field(default_factory=list)
    validation_keep_fractions: list = field(default_factory=list)
    spectral_weight: float = 0.1


@dataclass
class EvaluationConfig:
    audio_examples: int = 12
    bootstrap_samples: int = 1000
    snr_target_median: float = 30.0
    snr_target_p05: float = 20.0
    analysis_examples: int = 32
    keep_fractions: list = field(default_factory=list)
    speech_metrics: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    project_root: str = "."

    def path(self, value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else Path(self.project_root) / p

    def manifest(self, name):
        p = Path(name)
        return p if p.is_absolute() else self.path(self.data.manifest_dir) / p

    def to_dict(self):
        return asdict(self)


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def _read(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    parents = raw.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    result = {}
    for parent in parents:
        result = merge(result, _read(path.parent / parent, (*stack, path)))
    return merge(result, raw)


def from_dict(raw):
    raw = copy.deepcopy(raw)
    sections = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig, "evaluation": EvaluationConfig}
    unknown = set(raw) - {*sections, "project_root"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for section, cls in sections.items():
        values = raw.get(section, {})
        extra = set(values) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"Unknown {section} keys: {sorted(extra)}")
        raw[section] = cls(**values)
    cfg = Config(**raw)
    cfg.project_root = str(Path(cfg.project_root).resolve())
    validate(cfg)
    return cfg


def load_config(path):
    path = Path(path).resolve()
    raw = _read(path)
    root = next((p for p in path.parents if (p / "pyproject.toml").exists()), Path.cwd())
    raw.setdefault("project_root", str(root))
    return from_dict(raw)


@dataclass
class StudyConfig:
    """Settings for an experiment that does not require a trainable model."""

    name: str
    kind: str
    parameters: dict = field(default_factory=dict)
    runs_dir: str = "runs"
    project_root: str = "."

    def path(self, value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else Path(self.project_root) / p

    def to_dict(self):
        return asdict(self)


def load_study_config(path):
    """Reuse YAML inheritance/path rules without inventing a model or training job."""
    path = Path(path).resolve()
    raw = _read(path)
    extra = set(raw) - {f.name for f in fields(StudyConfig)}
    if extra:
        raise ValueError(f"Unknown study keys: {sorted(extra)}")
    for key in ("name", "kind"):
        if not isinstance(raw.get(key), str) or not re.fullmatch(r"[a-z][a-z0-9_]*", raw[key]):
            raise ValueError(f"Study {key} must use lower-case snake_case")
    if not isinstance(raw.get("parameters", {}), dict):
        raise ValueError("Study parameters must be a mapping")
    if not isinstance(raw.get("runs_dir", "runs"), str) or not raw.get("runs_dir", "runs"):
        raise ValueError("Study runs_dir must be a nonempty path string")
    root = next((p for p in path.parents if (p / "pyproject.toml").exists()), Path.cwd())
    raw.setdefault("project_root", str(root))
    cfg = StudyConfig(**raw)
    cfg.project_root = str(Path(cfg.project_root).expanduser().resolve())
    return cfg


def validate(c):
    if c.data.source not in {"synthetic", "librispeech", "vctk"}:
        raise ValueError("Unsupported dataset source")
    if c.data.vctk_mode not in {"external", "speaker_split"}:
        raise ValueError("VCTK mode must be external or speaker_split")
    if c.model.kind not in {"dct", "random", "pca", "orthogonal", "overcomplete", "grouped", "convolutional", "autoencoder", "sparse_autoencoder", "block_orthogonal"}:
        raise ValueError("Unsupported model kind")
    if c.data.sample_rate < 2000 or c.data.crop_seconds <= 0:
        raise ValueError("Invalid sampling rate or crop duration")
    if c.data.source == "synthetic" and c.data.sample_rate < 8000:
        raise ValueError("The configured synthetic parameter ranges require at least 8 kHz")
    m, t = c.model, c.train
    if c.data.librispeech_train_split not in {"train-clean-100", "train-clean-360", "train-other-500"}:
        raise ValueError("Unknown LibriSpeech training partition")
    if not math.isfinite(t.spectral_weight) or t.spectral_weight < 0:
        raise ValueError("spectral_weight must be finite and nonnegative")
    if t.keep_fractions or t.validation_keep_fractions or c.evaluation.keep_fractions:
        if m.kind != "sparse_autoencoder":
            raise ValueError("keep_fractions require sparse_autoencoder")
        if any(isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in [*t.keep_fractions, *t.validation_keep_fractions, *c.evaluation.keep_fractions]):
            raise ValueError("keep_fractions must be finite fractions between zero and one")
    if m.kind == "sparse_autoencoder":
        if m.activation not in {"leaky_relu", "identity"}:
            raise ValueError("sparse_autoencoder activation must be leaky_relu or identity")
        if m.groups != 1:
            raise ValueError("sparse_autoencoder uses one ungrouped representation")
        if not math.isfinite(m.keep_fraction) or not 0 <= m.keep_fraction <= 1:
            raise ValueError("keep_fraction must be between zero and one")
        if m.sample_budget is not None and (isinstance(m.sample_budget, bool) or not isinstance(m.sample_budget, int) or m.sample_budget < 0):
            raise ValueError("sample_budget must be a nonnegative integer")
        if m.code_budget or m.target_nmse is not None or t.functional_weight or t.group_weight or t.sparsity_weight:
            raise ValueError("sparse_autoencoder uses global keep_fraction/sample_budget without auxiliary sparsity or separation penalties")
    elif m.activation != "leaky_relu":
        raise ValueError("Alternative activation is only supported for sparse_autoencoder")
    if t.stage not in {"smoke", "reference", "screen", "main", "confirmation"}:
        raise ValueError("Unknown study stage")
    if m.kind == "block_orthogonal":
        if m.hop != m.frame_length:
            raise ValueError("block_orthogonal requires hop == frame_length (no overlap)")
        if m.initialization not in {"dct", "random", "identity"}:
            raise ValueError("Unknown block dictionary initialization")
        if not 0 <= m.keep_fraction <= 1 or m.sample_budget is not None and m.sample_budget < 0:
            raise ValueError("Require keep_fraction in [0, 1] and a nonnegative sample_budget")
        if m.target_nmse is not None and m.target_nmse < 0:
            raise ValueError("target_nmse must be nonnegative")
        if t.steps and m.target_nmse is not None:
            raise ValueError("Train with a fixed coefficient budget; use --nmse-targets for tolerance evaluation")
        if t.accumulation != 1:
            raise ValueError("SVD updates use batch_size directly; set accumulation to 1")
    elif not 0 < m.hop < m.frame_length:
        raise ValueError("Require 0 < hop < frame_length")
    if m.frame_length < 2 or m.groups < 1:
        raise ValueError("Require frame_length >= 2 and positive groups")
    if m.kind in {"autoencoder", "sparse_autoencoder"}:
        width = m.latent_width
    elif m.kind == "convolutional":
        width = m.conv_kernels
    elif m.kind == "overcomplete":
        width = m.atoms
    else:
        width = m.frame_length
    if width % m.groups:
        raise ValueError("Representation width must be divisible by groups")
    if len(m.channels) != 4 or min(m.channels) < 1:
        raise ValueError("Autoencoder requires four positive channel widths")
    if m.conv_length <= m.conv_hop or (m.conv_length - m.conv_hop) % 2:
        raise ValueError("Convolutional kernel minus hop must be positive and even")
    if min(t.batch_size, t.accumulation, t.val_interval, t.log_interval, t.checkpoint_interval, t.functional_interval, t.functional_probes) < 1:
        raise ValueError("Batch sizes and intervals must be positive")
    if t.steps < 0 or min(m.threshold, m.group_threshold, t.functional_weight, t.group_weight, t.sparsity_weight) < 0:
        raise ValueError("Steps and regularizers cannot be negative")
    if min(m.inference_steps, c.data.train_count, c.data.validation_count, c.data.test_count, c.evaluation.bootstrap_samples, t.val_examples) < 1:
        raise ValueError("Inference steps, dataset counts, and evaluation sample counts must be positive")
    if t.functional_weight and m.kind != "autoencoder":
        raise ValueError("Functional training is implemented for the nonlinear autoencoder")
    if m.kind != "autoencoder" and (t.group_weight or t.sparsity_weight):
        raise ValueError("Dictionary sparsity is controlled through model.threshold and model.group_threshold")
