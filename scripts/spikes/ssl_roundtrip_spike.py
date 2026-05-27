#!/usr/bin/env python
"""Spike: prove Parakeet-encoder -> NEST SSL -> reload-into-Parakeet round-trips.

This is the Stage-0 verification gate for the SSL continued-pretraining plan
(docs/superpowers/plans/2026-05-26-ssl-continued-pretraining.md, Task 0). It
proves we can:
  1. take Parakeet-tdt-0.6b-v3's FastConformer encoder config + weights,
  2. build a NEST-style SSL model whose encoder matches it exactly,
  3. load Parakeet's encoder weights into the SSL model (0 unexpected keys),
  4. run a few real SSL train steps (forward+backward) on synthetic audio,
  5. save the SSL .nemo, and load its encoder back into a fresh Parakeet ASR
     model (0 unexpected keys).

Discovered NeMo API (NeMo 2.7.3, installed in this repo's .venv):
  - SSL model class:  nemo.collections.asr.models.EncDecDenoiseMaskedTokenPredModel
        (the NEST denoising masked-token-prediction model; arXiv 2408.13106).
        Its base class EncDecMaskedTokenPredModel / SpeechEncDecSelfSupervisedModel
        also exist; the denoise variant is the one the design targets.
  - Example config (not shipped in the wheel; fetched from the matching release
    tag r2.7.0): examples/asr/conf/ssl/nest/nest_fast-conformer.yaml
        https://github.com/NVIDIA/NeMo/blob/r2.7.0/examples/asr/conf/ssl/nest/nest_fast-conformer.yaml
    The relevant model section of that config is inlined below as NEST_CFG so the
    spike is reproducible from a fresh clone (project reproducibility rule).

Key alignment fact: the NEST example uses an 80-mel preprocessor + d_model=512,
17-layer (Large) encoder. Parakeet-0.6b-v3 uses a 128-mel preprocessor +
d_model=1024, 24-layer encoder with use_bias=false / xscaling=false. We force the
SSL model's encoder cfg to Parakeet's `base.cfg.encoder` AND bump the SSL
preprocessor/quantizer/masking feat_in to 128 so the mel feature dim feeding the
encoder (encoder.feat_in) matches what the encoder weights expect.

Run:  HF_HUB_OFFLINE=0 python scripts/spikes/ssl_roundtrip_spike.py
Expect: ".. missing/unexpected" lines with unexpected==0 at both load points,
then "ROUND-TRIP OK".
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel

REF = "nvidia/parakeet-tdt-0.6b-v3"

# Inlined NEST model config (from examples/asr/conf/ssl/nest/nest_fast-conformer.yaml,
# tag r2.7.0). The encoder block is a placeholder; we overwrite model.encoder with
# Parakeet's encoder cfg at runtime. preprocessor.features is set to match Parakeet
# (128) below so the mel dim feeding the encoder matches the loaded weights.
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


def _write_tiny_manifest(d: Path, n: int = 4) -> Path:
    """Synthesize a few 16 kHz mono WAVs + a NeMo SSL manifest (audio + duration)."""
    sr = 16000
    rng = np.random.default_rng(0)
    manifest = d / "manifest.jsonl"
    with manifest.open("w") as mf:
        for i in range(n):
            dur = 2.0 + 0.5 * i  # 2.0 .. 3.5 s, all > min_duration=1.0
            wav = (0.05 * rng.standard_normal(int(dur * sr))).astype(np.float32)
            wp = d / f"{i:03d}.wav"
            sf.write(str(wp), wav, sr, subtype="PCM_16")
            mf.write(json.dumps({"audio_filepath": str(wp.resolve()), "duration": dur}) + "\n")
    return manifest


def main() -> None:
    torch.manual_seed(0)

    # 1. Load Parakeet; capture encoder cfg + state_dict.
    base = nemo_asr.models.ASRModel.from_pretrained(REF, map_location="cpu")
    enc_cfg = base.cfg.encoder
    print("encoder _target_:", enc_cfg.get("_target_"))
    print("d_model:", enc_cfg.get("d_model"), "n_layers:", enc_cfg.get("n_layers"),
          "feat_in:", enc_cfg.get("feat_in"), "subsampling_factor:", enc_cfg.get("subsampling_factor"))
    enc_sd = base.encoder.state_dict()
    enc_param_count = sum(v.numel() for v in enc_sd.values())
    print("encoder params:", enc_param_count)
    para_feat_in = int(enc_cfg.get("feat_in"))

    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        manifest = _write_tiny_manifest(dd, n=4)

        # 2. Build the SSL model with a matching encoder.
        ssl_cfg = OmegaConf.create(NEST_CFG)
        # Force the SSL encoder to be byte-for-byte Parakeet's encoder cfg.
        ssl_cfg.encoder = OmegaConf.create(OmegaConf.to_container(enc_cfg, resolve=True))
        # The mel feature dim feeding the encoder must match (encoder.feat_in == features).
        ssl_cfg.preprocessor.features = para_feat_in
        # ModelPT.__init__ calls setup_training_data, which rejects the ??? manifest;
        # provide a real manifest up front (validation_ds is omitted -> skipped).
        ssl_cfg.train_ds.manifest_filepath = str(manifest)
        # quantizer/masking feat_in interpolate from preprocessor.features, so they follow.
        OmegaConf.resolve(ssl_cfg)
        assert int(ssl_cfg.encoder.feat_in) == para_feat_in, ssl_cfg.encoder.feat_in

        ssl = EncDecDenoiseMaskedTokenPredModel(cfg=ssl_cfg)

        # 3. Load Parakeet encoder weights into the SSL encoder; assert 0 unexpected.
        missing, unexpected = ssl.encoder.load_state_dict(enc_sd, strict=False)
        print(f"[load #1 parakeet->ssl] missing: {len(missing)} unexpected: {len(unexpected)}")
        assert len(unexpected) == 0, ("unexpected SSL-encoder keys: " + repr(unexpected[:10]))
        # missing should also be 0 (encoder is identical); report if not.
        if missing:
            print("  WARNING missing keys (encoder not fully covered):", missing[:10])

        # 4. Run a few SSL train steps on synthetic audio. We drive the model's
        #    own forward + NEST MLM loss directly (training_step() reaches into
        #    self.trainer/self._optimizer for logging, which a manual loop lacks)
        #    -- this still exercises the full preprocessor->mask->encoder->decoder
        #    ->loss path and the encoder's backward.
        ssl.train()
        opt = torch.optim.AdamW(ssl.parameters(), lr=1e-4)
        steps = 0
        for batch in ssl._train_dl:
            batch = ssl.transfer_batch_to_device(batch, torch.device("cpu"), 0)
            opt.zero_grad()
            log_probs, encoded_len, masks, tokens = ssl.forward(
                input_signal=batch.audio,
                input_signal_length=batch.audio_len,
                noise_signal=batch.noise,
                noise_signal_length=batch.noise_len,
                noisy_input_signal=batch.noisy_audio,
                noisy_input_signal_length=batch.noisy_audio_len,
                apply_mask=True,
            )
            loss = ssl.loss(
                masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len
            )
            loss.backward()
            opt.step()
            print(f"  step {steps}: loss={float(loss):.4f}")
            steps += 1
            if steps >= 6:
                break
        assert steps > 0, "no SSL train steps ran"

        # 5. Save the SSL .nemo, reload its encoder into a FRESH Parakeet.
        ssl_path = dd / "ssl.nemo"
        ssl.save_to(str(ssl_path))
        print("saved SSL model to", ssl_path, "exists:", ssl_path.exists())

        tgt = nemo_asr.models.ASRModel.from_pretrained(REF, map_location="cpu")
        reloaded = EncDecDenoiseMaskedTokenPredModel.restore_from(str(ssl_path), map_location="cpu")
        m2, u2 = tgt.encoder.load_state_dict(reloaded.encoder.state_dict(), strict=False)
        print(f"[load #2 ssl->parakeet] missing: {len(m2)} unexpected: {len(u2)}")
        assert len(u2) == 0, ("unexpected Parakeet-encoder keys: " + repr(u2[:10]))
        assert len(m2) == 0, ("missing Parakeet-encoder keys: " + repr(m2[:10]))

    print("ROUND-TRIP OK")


if __name__ == "__main__":
    main()
