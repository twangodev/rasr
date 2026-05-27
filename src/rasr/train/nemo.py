from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rasr.train.config import TrainConfig
from rasr.train.manifest import build_manifest


def _concat_manifests(paths: list[Path], weights: list[float], out: Path) -> Path:
    """Combine multiple manifests into one, repeating each `weight` times.

    NeMo doesn't natively weight non-tarred manifests, so we materialize the
    weighting by line repetition. Round to nearest int (min 1).
    """
    with out.open("w") as f:
        for path, w in zip(paths, weights, strict=True):
            reps = max(1, round(w))
            lines = path.read_text().splitlines()
            for _ in range(reps):
                for line in lines:
                    if line.strip():
                        f.write(line + "\n")
    return out


def run(cfg: TrainConfig) -> Path:
    """Execute a training run end-to-end. Returns the path to the saved .nemo."""
    import nemo.collections.asr as nemo_asr
    import lightning.pytorch as pl
    from omegaconf import OmegaConf
    from lightning.pytorch.callbacks import ModelCheckpoint

    if cfg.model.scheme != "parakeet":
        raise NotImplementedError(
            f"only 'parakeet' scheme is wired for training, got {cfg.model.scheme!r}"
        )

    cache_dir = Path("data/cache/train")
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rasr.train] building train manifests ({len(cfg.data.train)} dataset(s))")
    train_paths = [
        build_manifest(ds, cfg.data.audio, cache_dir) for ds in cfg.data.train
    ]
    print(f"[rasr.train] building val manifests ({len(cfg.data.validation)} dataset(s))")
    val_paths = [
        build_manifest(ds, cfg.data.audio, cache_dir) for ds in cfg.data.validation
    ]

    if len(train_paths) == 1 and cfg.data.train[0].weight == 1.0:
        train_manifest = train_paths[0]
    else:
        combined = output_dir / "train_manifest.combined.jsonl"
        train_manifest = _concat_manifests(
            train_paths, [d.weight for d in cfg.data.train], combined
        )
    val_manifest = val_paths[0]
    if len(val_paths) > 1:
        print("[rasr.train] multiple val datasets given; using the first only")

    n_train = sum(1 for _ in train_manifest.open())
    n_val = sum(1 for _ in val_manifest.open())
    print(f"[rasr.train] train manifest: {train_manifest} ({n_train} rows)")
    print(f"[rasr.train] val manifest:   {val_manifest} ({n_val} rows)")

    print(f"[rasr.train] loading base model: {cfg.model.ref}")
    model = nemo_asr.models.ASRModel.from_pretrained(cfg.model.ref)

    if cfg.model.init_encoder_from:
        from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel
        try:
            src = EncDecDenoiseMaskedTokenPredModel.restore_from(
                cfg.model.init_encoder_from, map_location="cpu"
            )
        except Exception:
            src = nemo_asr.models.ASRModel.restore_from(
                cfg.model.init_encoder_from, map_location="cpu"
            )
        missing, unexpected = model.encoder.load_state_dict(
            src.encoder.state_dict(), strict=False
        )
        print(
            f"[rasr.train] loaded SSL encoder from {cfg.model.init_encoder_from} "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )
        assert not unexpected, f"unexpected encoder keys: {unexpected[:5]}"
        del src

    sa = cfg.augmentation.spec_augment
    sp = cfg.augmentation.speed_perturb
    bp = cfg.augmentation.bandpass
    augmentor = {}
    if sp.enabled:
        augmentor["speed"] = {
            "prob": sp.prob,
            "sr": cfg.data.audio.sample_rate,
            "min_speed_rate": sp.min_rate,
            "max_speed_rate": sp.max_rate,
            "resample_type": "kaiser_fast",
        }
    if bp.enabled:
        from rasr.train.augment import register as _register_bandpass

        _register_bandpass()
        augmentor["bandpass"] = {
            "prob": bp.prob,
            "low_hz": bp.low_hz,
            "high_hz": bp.high_hz,
            "sr": cfg.data.audio.sample_rate,
        }

    train_ds_cfg = OmegaConf.create(
        {
            "manifest_filepath": str(train_manifest),
            "sample_rate": cfg.data.audio.sample_rate,
            "batch_size": cfg.trainer.batch_size,
            "shuffle": True,
            "num_workers": cfg.trainer.num_workers,
            "pin_memory": True,
            "max_duration": cfg.data.audio.max_duration,
            "min_duration": cfg.data.audio.min_duration,
            "is_tarred": False,
            "trim_silence": False,
            "augmentor": augmentor,
        }
    )
    val_ds_cfg = OmegaConf.create(
        {
            "manifest_filepath": str(val_manifest),
            "sample_rate": cfg.data.audio.sample_rate,
            "batch_size": cfg.trainer.batch_size,
            "shuffle": False,
            "num_workers": min(4, cfg.trainer.num_workers),
            "pin_memory": True,
        }
    )
    model.setup_training_data(train_ds_cfg)
    model.setup_validation_data(val_ds_cfg)

    # Override SpecAugment on the model's preprocessor.
    if hasattr(model, "spec_augmentation") and model.spec_augmentation is not None:
        model.spec_augmentation.freq_masks = sa.freq_masks
        model.spec_augmentation.time_masks = sa.time_masks
        model.spec_augmentation.freq_width = sa.freq_width
        model.spec_augmentation.time_width = sa.time_width

    optim_cfg = OmegaConf.create(
        {
            "name": cfg.optimizer.name,
            "lr": cfg.optimizer.lr,
            "betas": list(cfg.optimizer.betas),
            "weight_decay": cfg.optimizer.weight_decay,
            "sched": {
                "name": cfg.scheduler.name,
                "warmup_steps": cfg.scheduler.warmup_steps,
                "min_lr": cfg.scheduler.min_lr,
            },
        }
    )
    model.cfg.optim = optim_cfg
    model.setup_optimization(optim_cfg)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="step{step:06d}-wer{val_wer:.4f}",
        monitor=cfg.output.monitor,
        mode=cfg.output.mode,
        save_top_k=cfg.output.save_top_k,
        auto_insert_metric_name=False,
        save_last=True,
    )

    trainer = pl.Trainer(
        devices=cfg.trainer.devices,
        accelerator="gpu",
        precision=cfg.trainer.precision,
        max_steps=cfg.trainer.max_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=20,
        callbacks=[ckpt_cb],
        enable_progress_bar=True,
    )

    print(f"[rasr.train] starting fit: max_steps={cfg.trainer.max_steps}")
    trainer.fit(model)

    final_path = output_dir / "final.nemo"
    model.save_to(str(final_path))
    print(f"[rasr.train] saved: {final_path}")
    return final_path
