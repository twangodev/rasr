from __future__ import annotations

from pathlib import Path

from rasr.train.config import AudioCfg, SSLDatasetCfg, SSLTrainConfig
from rasr.train.ssl_manifest import build_ssl_manifest

# Inlined NEST model config (from examples/asr/conf/ssl/nest/nest_fast-conformer.yaml,
# tag r2.7.0; NOT shipped in the pip wheel). The `encoder` block is a placeholder we
# overwrite with the Parakeet encoder cfg at runtime, and `preprocessor.features` is
# forced to 128 (Parakeet's feat_in) so the mel dim feeding the encoder matches the
# loaded weights. quantizer/masking feat_in interpolate from preprocessor.features.
# See scripts/spikes/ssl_roundtrip_spike.py for the proven construction.
NEST_CFG = """
sample_rate: 16000
num_classes: 8192
num_books: 1
code_dim: 16
squeeze_single: false
mask_position: pre_conv

train_ds:
  manifest_filepath: ???
  noise_manifest: null
  sample_rate: ${sample_rate}
  batch_size: 2
  shuffle: true
  num_workers: 0
  pin_memory: false
  use_start_end_token: false
  trim_silence: false
  max_duration: 60.0
  min_duration: 1.0
  drop_last: true
  is_concat: false
  batch_augmentor:
    _target_: nemo.collections.asr.modules.ssl_modules.MultiSpeakerNoiseAugmentation
    prob: 0.5
    noise_ratio: 0.5
    min_r_speech: -5.0
    max_r_speech: 5.0
    min_r_noise: -5.0
    max_r_noise: 20.0
    min_mix_rate: 0.5
    max_mix_rate: 0.5
    min_num_segments: 1
    max_num_segments: 1
    min_num_speakers: 1
    max_num_speakers: 1

preprocessor:
  _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
  sample_rate: ${sample_rate}
  normalize: per_feature
  window_size: 0.025
  window_stride: 0.01
  window: hann
  features: 128
  n_fft: 512
  log: true
  frame_splicing: 1
  dither: 0.00001
  pad_to: 16
  pad_value: 0.0

spec_augment:
  _target_: nemo.collections.asr.modules.SpectrogramAugmentation
  freq_masks: 0
  time_masks: 0
  freq_width: 27
  time_width: 0.05

masking:
  _target_: nemo.collections.asr.modules.RandomBlockMasking
  block_size: 40
  mask_prob: 0.01
  feat_in: ${preprocessor.features}
  freeze: true
  allow_overlap: true

quantizer:
  _target_: nemo.collections.asr.modules.RandomProjectionVectorQuantizer
  feat_in: ${preprocessor.features}
  code_dim: ${code_dim}
  num_books: ${num_books}
  num_classes: ${num_classes}
  dist_fn: l2
  freeze: true
  squeeze_single: ${squeeze_single}
  combine_time_steps: ${encoder.subsampling_factor}

encoder:
  _target_: nemo.collections.asr.modules.ConformerEncoder
  feat_in: ${preprocessor.features}
  feat_out: -1
  n_layers: 17
  d_model: 512
  subsampling: dw_striding
  subsampling_factor: 8
  subsampling_conv_channels: 256
  ff_expansion_factor: 4
  self_attention_model: rel_pos
  n_heads: 8
  conv_kernel_size: 9
  conv_norm_type: batch_norm

decoder:
  _target_: nemo.collections.asr.modules.MultiSoftmaxDecoder
  feat_in: ${encoder.d_model}
  num_classes: ${num_classes}
  num_decoders: ${num_books}
  squeeze_single: ${squeeze_single}
  use_bias: true

loss:
  _target_: nemo.collections.asr.losses.MultiMLMLoss
  combine_time_steps: ${encoder.subsampling_factor}
  mask_threshold: 0.8
  num_decoders: ${num_books}
  squeeze_single: ${squeeze_single}

optim:
  name: adamw
  lr: 1e-4
  betas: [0.9, 0.98]
  weight_decay: 1e-3
"""


def run(cfg: SSLTrainConfig) -> Path:
    """Continue-pretrain a Parakeet encoder with NEST SSL. Returns saved .nemo path."""
    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    from lightning.pytorch.callbacks import ModelCheckpoint
    from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel
    from omegaconf import OmegaConf

    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build the SSL manifest (no text) from the configured source.
    print(f"[rasr.train.ssl] building SSL manifest from {cfg.ssl.source}")
    manifest = build_ssl_manifest(
        SSLDatasetCfg(
            source=cfg.ssl.source,
            limit=cfg.ssl.limit,
            min_db=cfg.ssl.min_db,
            segment=cfg.ssl.segment,
            segment_top_db=cfg.ssl.segment_top_db,
            segment_min_s=cfg.ssl.segment_min_s,
            segment_max_s=cfg.ssl.segment_max_s,
            segment_pad_s=cfg.ssl.segment_pad_s,
        ),
        cfg.audio,
        Path("data/cache/ssl"),
    )
    n_rows = sum(1 for _ in manifest.open())
    print(f"[rasr.train.ssl] SSL manifest: {manifest} ({n_rows} rows)")

    # 2. Load the Parakeet ref to source the encoder cfg (+ weights below).
    if cfg.ssl.init_encoder_from is None:
        raise ValueError(
            "cfg.ssl.init_encoder_from must be set: NEST SSL needs a Parakeet "
            "encoder cfg to build a matching SSL encoder."
        )
    print(f"[rasr.train.ssl] loading encoder cfg/weights from {cfg.ssl.init_encoder_from}")
    base = nemo_asr.models.ASRModel.from_pretrained(
        cfg.ssl.init_encoder_from, map_location="cpu"
    )
    enc_cfg = base.cfg.encoder
    para_feat_in = int(enc_cfg.get("feat_in"))

    # 3. Build the SSL model with an encoder matching Parakeet's exactly.
    ssl_cfg = OmegaConf.create(NEST_CFG)
    ssl_cfg.encoder = OmegaConf.create(OmegaConf.to_container(enc_cfg, resolve=True))
    ssl_cfg.preprocessor.features = para_feat_in  # encoder.feat_in == mel features
    ssl_cfg.sample_rate = cfg.audio.sample_rate
    ssl_cfg.train_ds.manifest_filepath = str(manifest)  # real manifest BEFORE init
    ssl_cfg.train_ds.batch_size = cfg.trainer.batch_size
    ssl_cfg.train_ds.num_workers = cfg.trainer.num_workers
    ssl_cfg.train_ds.sample_rate = cfg.audio.sample_rate
    # Keep the NeMo dataloader's duration filter in sync with the manifest cap.
    # Otherwise NEST_CFG's hardcoded max_duration silently re-drops every clip the
    # manifest kept (e.g. TartanAviation recordings run minutes-long, far past 60s).
    ssl_cfg.train_ds.max_duration = cfg.audio.max_duration
    ssl_cfg.train_ds.min_duration = cfg.audio.min_duration
    OmegaConf.resolve(ssl_cfg)
    assert int(ssl_cfg.encoder.feat_in) == para_feat_in, ssl_cfg.encoder.feat_in

    model = EncDecDenoiseMaskedTokenPredModel(cfg=ssl_cfg)

    # 4. Seed the SSL encoder with Parakeet's weights (0 unexpected proven by spike).
    missing, unexpected = model.encoder.load_state_dict(
        base.encoder.state_dict(), strict=False
    )
    print(
        f"[rasr.train.ssl] encoder init: missing={len(missing)} unexpected={len(unexpected)}"
    )
    assert len(unexpected) == 0, "unexpected SSL-encoder keys: " + repr(unexpected[:10])
    if missing:
        print("[rasr.train.ssl] WARNING missing encoder keys:", missing[:10])
    del base

    # 5. Optimizer/scheduler from cfg (mirror nemo.py).
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

    # 6. NEST is SSL: no validation set / no val-WER. Checkpoint on train_loss
    #    plus save_last so a run is always resumable.
    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="step{step:06d}-loss{train_loss:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=cfg.output.save_top_k,
        auto_insert_metric_name=False,
        save_last=True,
    )

    trainer = pl.Trainer(
        devices=cfg.trainer.devices,
        accelerator="gpu",
        precision=cfg.trainer.precision,
        max_steps=cfg.trainer.max_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=20,
        callbacks=[ckpt_cb],
        enable_progress_bar=True,
    )

    print(f"[rasr.train.ssl] starting fit: max_steps={cfg.trainer.max_steps}")
    trainer.fit(model)

    final_path = output_dir / "final.nemo"
    model.save_to(str(final_path))
    print(f"[rasr.train.ssl] saved: {final_path}")
    return final_path
