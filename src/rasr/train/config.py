from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field


class ModelCfg(BaseModel):
    scheme: str
    ref: str
    language: str | None = None


class DatasetCfg(BaseModel):
    dataset: str
    weight: float = 1.0
    limit: int | None = None


class AudioCfg(BaseModel):
    sample_rate: int = 16000
    max_duration: float = 18.0
    min_duration: float = 0.1


class DataCfg(BaseModel):
    train: list[DatasetCfg]
    validation: list[DatasetCfg]
    audio: AudioCfg = Field(default_factory=AudioCfg)
    use_lhotse: bool = True


class OptimizerCfg(BaseModel):
    name: str = "adamw"
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 1e-3


class SchedulerCfg(BaseModel):
    name: str = "CosineAnnealing"
    warmup_steps: int = 5000
    min_lr: float = 1e-6


class SpecAugmentCfg(BaseModel):
    freq_masks: int = 2
    time_masks: int = 10
    freq_width: int = 27
    time_width: float = 0.05


class SpeedPerturbCfg(BaseModel):
    enabled: bool = True
    prob: float = 0.5
    min_rate: float = 0.95
    max_rate: float = 1.05


class NoiseCfg(BaseModel):
    enabled: bool = False


class BandpassCfg(BaseModel):
    enabled: bool = False
    low_hz: int = 300
    high_hz: int = 3400
    prob: float = 1.0


class AugmentationCfg(BaseModel):
    spec_augment: SpecAugmentCfg = Field(default_factory=SpecAugmentCfg)
    speed_perturb: SpeedPerturbCfg = Field(default_factory=SpeedPerturbCfg)
    noise: NoiseCfg = Field(default_factory=NoiseCfg)
    bandpass: BandpassCfg = Field(default_factory=BandpassCfg)


class TrainerCfg(BaseModel):
    max_steps: int
    batch_size: int
    num_workers: int = 16
    precision: str = "bf16-mixed"
    devices: int = 1
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    val_check_interval: int = 1000


class OutputCfg(BaseModel):
    dir: str
    save_top_k: int = 3
    monitor: str = "val_wer"
    mode: str = "min"


class TrainConfig(BaseModel):
    name: str
    model: ModelCfg
    data: DataCfg
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    scheduler: SchedulerCfg = Field(default_factory=SchedulerCfg)
    augmentation: AugmentationCfg = Field(default_factory=AugmentationCfg)
    trainer: TrainerCfg
    output: OutputCfg


def _merge_with_defaults(path: Path, root: Path) -> DictConfig:
    """Resolve `defaults: [...]` recursively, then layer the current file on top.

    Each entry in `defaults` is a path relative to `root` (the configs/train
    directory), without the .yaml suffix. Layers are merged in list order; the
    current file's keys override anything in its defaults.
    """
    raw: DictConfig = OmegaConf.load(path)  # type: ignore[assignment]
    defaults = raw.pop("defaults", [])
    merged = OmegaConf.create({})
    for entry in defaults:
        dep_path = (root / entry).with_suffix(".yaml")
        merged = OmegaConf.merge(merged, _merge_with_defaults(dep_path, root))
    merged = OmegaConf.merge(merged, raw)
    return merged  # type: ignore[return-value]


def load_train_config(
    path: Path,
    overrides: list[str] | None = None,
) -> TrainConfig:
    """Load a training YAML, resolve `defaults:`, apply CLI overrides, validate.

    `overrides` is a list of Hydra-style `key.path=value` strings (e.g.
    `trainer.max_steps=100000`).
    """
    path = Path(path).resolve()
    root = _find_configs_root(path)
    merged = _merge_with_defaults(path, root)
    if overrides:
        merged = OmegaConf.merge(
            merged, OmegaConf.from_dotlist(list(overrides))
        )
    OmegaConf.resolve(merged)
    return TrainConfig.model_validate(OmegaConf.to_container(merged, resolve=True))


def _find_configs_root(path: Path) -> Path:
    """Walk up until we find a directory named `train` (under `configs/`)."""
    for parent in path.parents:
        if parent.name == "train" and parent.parent.name == "configs":
            return parent
    raise ValueError(
        f"Could not locate configs/train/ ancestor of {path}; "
        "training configs must live under configs/train/"
    )
