#!/usr/bin/env python
"""Diagnostic: does base Parakeet's encoder find REAL ATC harder than SYNTHETIC?

Necessary-condition check for whether continued SSL pretraining (CPT) on real ATC
audio could help. We build a NEST masked-denoising SSL model (NeMo's
EncDecDenoiseMaskedTokenPredModel) seeded with Parakeet-tdt-0.6b-v3's FastConformer
encoder weights -- the exact construction proven in
scripts/spikes/ssl_roundtrip_spike.py -- and measure its NEST (masked token
prediction) loss on two domains:

  * REAL ATC   = TartanAviation sample on disk (data/tartanaviation/sample/**/*.wav),
                 energy-VAD segmented into utterance chunks.
  * SYNTHETIC  = the cached Higgs dataset
                 (hf:twangodev/radiotalk-us-audio-higgs-noisy:train), segmented.

Interpretation: a substantially HIGHER NEST loss on real ATC means the encoder
models real ATC acoustics worse than synthetic -- an acoustic gap CPT could close.
Similar losses => little acoustic headroom => CPT unlikely to help much.

Fairness: both manifests are restricted to clips of similar duration (2-10 s) and
the SAME number of clips (min of the two, capped). torch+numpy seeds are reset
before each domain loop so the NEST masking pattern is comparable across domains.
The loss path mirrors ssl_roundtrip_spike.py: model forward + NEST MLM loss, in
eval()/no_grad(). To isolate acoustic modeling we feed noisy_input_signal =
input_signal (no added noise/augmentation), so the loss measures the encoder's
ability to predict masked clean targets.

IMPORTANT CAVEAT (found while running this): a NEST model built by transplanting
ONLY Parakeet's encoder leaves the decoder (MultiSoftmaxDecoder) and the
RandomProjectionVectorQuantizer targets UNTRAINED. The masked-token-prediction
head therefore predicts ~uniformly over the 8192-entry codebook, so the loss
pins at the chance floor ln(8192) = 9.011 for BOTH domains and does NOT reflect
how well the encoder models the audio. The script prints this caveat when both
losses are at chance. To get a meaningful acoustic-gap signal you must first
train the decoder against the frozen quantizer on pooled audio, THEN compare the
per-domain loss; or use a decoder-free probe (e.g. encoder-feature divergence).

Run:  HF_HUB_OFFLINE=1 python scripts/spikes/ssl_gap_diagnostic.py
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel

from rasr.train.config import AudioCfg, SSLDatasetCfg
from rasr.train.ssl_manifest import build_ssl_manifest

# Reuse the proven NEST config from the round-trip spike (see that file for the
# provenance: examples/asr/conf/ssl/nest/nest_fast-conformer.yaml, tag r2.7.0).
# The encoder block is a placeholder overwritten with Parakeet's encoder cfg.
# Loaded by file path because scripts/ is not an importable package.
_SPIKE = Path(__file__).resolve().parent / "ssl_roundtrip_spike.py"
_spec = importlib.util.spec_from_file_location("ssl_roundtrip_spike", _SPIKE)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
NEST_CFG = _mod.NEST_CFG
REF = _mod.REF

CACHE = Path("data/cache/ssl_diag")
SEED = 0
DUR_LO, DUR_HI = 2.0, 10.0  # fair-comparison duration window (seconds)
MAX_CLIPS = 150


def build_model() -> EncDecDenoiseMaskedTokenPredModel:
    """Construct the NEST model with Parakeet's encoder weights (per the spike)."""
    base = nemo_asr.models.ASRModel.from_pretrained(REF, map_location="cpu")
    enc_cfg = base.cfg.encoder
    enc_sd = base.encoder.state_dict()
    para_feat_in = int(enc_cfg.get("feat_in"))

    ssl_cfg = OmegaConf.create(NEST_CFG)
    ssl_cfg.encoder = OmegaConf.create(OmegaConf.to_container(enc_cfg, resolve=True))
    ssl_cfg.preprocessor.features = para_feat_in
    # The inlined recipe's masking (block_size=40, mask_prob=0.01) is tuned for
    # large batched training; on single short clips the post-subsampling sequence
    # is short (~60 frames) so a 0.01 per-position block prob frequently masks
    # ZERO blocks, and MultiMLMLoss does mean() over an empty masked set -> NaN.
    # We use the NEST module's own defaults (block_size=8, mask_prob=0.5) so every
    # clip gets a non-empty mask. This is applied identically to both domains, so
    # the real-vs-synthetic comparison stays fair.
    ssl_cfg.masking.block_size = 8
    ssl_cfg.masking.mask_prob = 0.5
    # ModelPT.__init__ calls setup_training_data; the ??? manifest would be rejected.
    # We never use the train dataloader here (we feed clips manually), but we must
    # hand it a real, tiny manifest so construction succeeds.
    tmp_manifest = CACHE / "_ctor_manifest.jsonl"
    tmp_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_wav = CACHE / "_ctor.wav"
    if not tmp_wav.exists():
        sf.write(str(tmp_wav), np.zeros(16000 * 2, dtype=np.float32), 16000, subtype="PCM_16")
    tmp_manifest.write_text(
        json.dumps({"audio_filepath": str(tmp_wav.resolve()), "duration": 2.0}) + "\n"
    )
    ssl_cfg.train_ds.manifest_filepath = str(tmp_manifest)
    OmegaConf.resolve(ssl_cfg)
    assert int(ssl_cfg.encoder.feat_in) == para_feat_in, ssl_cfg.encoder.feat_in

    ssl = EncDecDenoiseMaskedTokenPredModel(cfg=ssl_cfg)
    missing, unexpected = ssl.encoder.load_state_dict(enc_sd, strict=False)
    print(f"[parakeet->ssl encoder] missing: {len(missing)} unexpected: {len(unexpected)}")
    assert len(unexpected) == 0, ("unexpected SSL-encoder keys: " + repr(unexpected[:10]))
    if missing:
        print("  WARNING missing keys (encoder not fully covered):", missing[:10])
    return ssl


def load_manifest_clips(manifest: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append((r["audio_filepath"], float(r["duration"])))
    return rows


def select_fair(
    real: list[tuple[str, float]], synth: list[tuple[str, float]]
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Restrict both to [DUR_LO, DUR_HI] and take the same N (min, capped)."""
    rf = [c for c in real if DUR_LO <= c[1] <= DUR_HI]
    sf_ = [c for c in synth if DUR_LO <= c[1] <= DUR_HI]
    # Deterministic, seed-stable subsample so both draws are reproducible.
    rng = random.Random(SEED)
    rng.shuffle(rf)
    rng2 = random.Random(SEED)
    rng2.shuffle(sf_)
    n = min(len(rf), len(sf_), MAX_CLIPS)
    return rf[:n], sf_[:n]


def _seed_all() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def mean_nest_loss(
    ssl: EncDecDenoiseMaskedTokenPredModel,
    clips: list[tuple[str, float]],
    device: torch.device,
) -> float:
    """Average NEST MLM loss over clips. Resets seeds first so masking is comparable.

    Mirrors ssl_roundtrip_spike.py's loss path (forward + self.loss), in
    eval()/no_grad(), one clip per forward. noisy_input_signal = input_signal so
    the encoder sees the un-augmented audio and the loss isolates acoustic modeling.
    """
    _seed_all()
    total, count, skipped = 0.0, 0, 0
    with torch.no_grad():
        for path, _dur in clips:
            arr, sr = sf.read(path, dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            assert sr == 16000, f"expected 16k, got {sr} for {path}"
            audio = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(device)
            audio_len = torch.tensor([arr.shape[0]], dtype=torch.long, device=device)
            log_probs, encoded_len, masks, tokens = ssl.forward(
                input_signal=audio,
                input_signal_length=audio_len,
                noisy_input_signal=audio,
                noisy_input_signal_length=audio_len,
                apply_mask=True,
            )
            loss = ssl.loss(
                masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len
            )
            lv = float(loss)
            # Skip the rare clip that still masks nothing (empty-set mean -> NaN).
            if lv != lv:
                skipped += 1
                continue
            total += lv
            count += 1
    if skipped:
        print(f"  (skipped {skipped} clips with empty mask -> NaN loss)")
    assert count > 0, "no clips scored"
    return total / count


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ssl = build_model()
    ssl.eval().to(device)

    audio_cfg = AudioCfg(max_duration=30.0, min_duration=0.5)

    print("building REAL ATC manifest (TartanAviation sample, VAD-segmented)...")
    real_manifest = build_ssl_manifest(
        SSLDatasetCfg(
            source="local:data/tartanaviation/sample/**/*.wav",
            min_db=-20.0,
            segment=True,
        ),
        audio_cfg,
        CACHE,
    )
    print("building SYNTHETIC manifest (Higgs cached, VAD-segmented)...")
    synth_manifest = build_ssl_manifest(
        SSLDatasetCfg(
            source="hf:twangodev/radiotalk-us-audio-higgs-noisy:train",
            limit=400,
            min_db=-20.0,
            segment=True,
        ),
        audio_cfg,
        CACHE,
    )

    real_all = load_manifest_clips(real_manifest)
    synth_all = load_manifest_clips(synth_manifest)
    print(f"raw clips -> real: {len(real_all)}  synth: {len(synth_all)}")

    real_clips, synth_clips = select_fair(real_all, synth_all)
    n = len(real_clips)
    if n == 0:
        raise RuntimeError(
            "No clips in the fair duration window for one/both domains; "
            f"real_in_window={sum(DUR_LO <= c[1] <= DUR_HI for c in real_all)} "
            f"synth_in_window={sum(DUR_LO <= c[1] <= DUR_HI for c in synth_all)}"
        )
    rd = np.mean([c[1] for c in real_clips])
    sd = np.mean([c[1] for c in synth_clips])
    print(f"fair set: N={n} per domain | mean dur real={rd:.2f}s synth={sd:.2f}s "
          f"(window {DUR_LO}-{DUR_HI}s)")

    real_loss = mean_nest_loss(ssl, real_clips, device)
    synth_loss = mean_nest_loss(ssl, synth_clips, device)

    delta = real_loss - synth_loss
    ratio = real_loss / synth_loss if synth_loss else float("nan")

    # NEST predicts over num_classes codebook entries; a random/untrained
    # prediction head sits at the chance floor ln(num_classes). If both domains
    # land here, the loss carries NO acoustic signal (see caveat below).
    chance = float(np.log(int(ssl.cfg.num_classes)))

    print("\n=== NEST-loss acoustic gap diagnostic ===")
    print(f"REAL ATC  (TartanAviation): mean NEST loss = {real_loss:.4f}  (N={n})")
    print(f"SYNTHETIC (Higgs noisy)   : mean NEST loss = {synth_loss:.4f}  (N={n})")
    print(f"delta (real - synth) = {delta:+.4f}   ratio (real/synth) = {ratio:.3f}x")
    print(f"chance floor ln(num_classes={int(ssl.cfg.num_classes)}) = {chance:.4f}")
    near_chance = abs(real_loss - chance) < 0.05 and abs(synth_loss - chance) < 0.05
    if near_chance:
        print(
            "CAVEAT: both losses sit at the chance floor. The NEST decoder and "
            "RandomProjectionVectorQuantizer here are UNTRAINED (only the encoder "
            "was transplanted from Parakeet), so the masked-token-prediction head "
            "predicts ~uniformly regardless of input. This NEST-loss proxy is "
            "therefore UNINFORMATIVE about the encoder's acoustic fit; the "
            "domain comparison below is not meaningful. A valid version would "
            "first train the decoder against the frozen quantizer (a few hundred "
            "steps on pooled audio), then compare per-domain loss."
        )
    if ratio >= 1.15:
        verdict = ("GAP: real ATC is substantially harder -> acoustic headroom; "
                   "CPT on real ATC is plausibly worth trying.")
    elif ratio <= 0.95:
        verdict = ("INVERTED: synthetic is harder than real -> no real-ATC acoustic "
                   "gap by this measure; CPT unlikely to help on acoustics.")
    else:
        verdict = ("SIMILAR: little acoustic headroom between domains -> CPT on real "
                   "ATC unlikely to help much (by this NEST-loss proxy).")
    print(f"interpretation: {verdict}")


if __name__ == "__main__":
    main()
