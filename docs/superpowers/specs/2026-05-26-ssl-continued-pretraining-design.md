# SSL continued-pretraining for rasr — design

## Goal

Adapt `nvidia/parakeet-tdt-0.6b-v3`'s encoder to real US-ATC VHF acoustics via
**continued self-supervised pretraining (CPT)** on unlabeled TartanAviation
audio, *before* the supervised finetune on synthetic Higgs data. Hypothesis:
encoder-level acoustic adaptation closes the synthetic-vs-real gap that
synthetic supervised data alone cannot.

This produces the planned "final" rasr model. It does not replace the supervised
finetune — it precedes it. The lr/warmup tuned in
`docs/higgs-lr-warmup-sweep.md` (lr=2.5e-5, warmup~500) applies to the
supervised stage that follows CPT.

## Background / decisions already made

- **Corpus: TartanAviation** (CMU AirLab, arXiv 2403.03372, CC BY 4.0, free at
  theairlab.org/tartanaviation). 41,823 ADS-B-triggered WAV clips, 44.1 kHz,
  ~3,375 hr raw, **~478 hr clearing the >=-20 dB speech-quality threshold**
  (use the filtered subset). Already segmented (>1 s), fully unlabeled. Real US
  terminal-area ATC — matches the US synthetic finetune domain.
- **Method: NEST** (NeMo FastConformer-native SSL: random-projection masked
  denoising + noisy-speech augmentation). Scripts in
  `examples/asr/speech_pretraining`.
- **Strategy A — adapt Parakeet's own ~600M encoder.** Initialize the SSL
  model's FastConformer encoder from Parakeet, CPT it, then load the adapted
  encoder back into Parakeet for finetuning. The public NEST checkpoints
  (`ssl_en_nest_large_v1.0`, 115M) are shape-incompatible with Parakeet-0.6b,
  so they are *not* used as weights — only the method is borrowed.
- Reproducibility: TartanAviation is freely downloadable; prep must work from a
  fresh clone (no local-only paths), consistent with the project rule.

## Stages / components

### Stage 0 — Verification spike (DO FIRST; load-bearing risk)

Before any long run, prove the round-trip works on a tiny scale:
1. Build the SSL model with an encoder config matching Parakeet-0.6b-v3's
   encoder; initialize encoder weights from the Parakeet `.nemo`.
2. Run ~50 SSL steps on a handful of clips; save.
3. Load the saved encoder back into a Parakeet ASR model
   (`++init_from_pretrained.ssl.include=["encoder"]`) and confirm the weights
   load with no shape errors and inference runs.

If the encoder configs cannot be aligned cleanly, fall back options:
(a) freeze-encoder CPT variants, or (b) drop to a NEST-large-sized encoder
(115M) and accept a lower ceiling. Decide only if the spike fails.

### Stage 1 — Unlabeled data prep (`rasr`)

TartanAviation -> NeMo **SSL manifest** (`audio_filepath` + `duration`, no
`text`). Resample 44.1k->16k mono; keep the quality-filtered subset. This is a
no-text variant of the existing `build_manifest` (reuse caching + the offline
hub-cache fallback). New `DatasetCfg`-style spec for an unlabeled source.

### Stage 2 — Continued SSL pretraining (new `rasr` stage)

NEST masked-denoising on the Parakeet-initialized FastConformer encoder over
the TartanAviation manifest. Continued (not from-scratch): ~tens of thousands of
steps, single 6000-Pro, bf16-mixed. New recipe `configs/train/.../*-ssl.yaml`
and a `rasr` entry that wraps NeMo's SSL training. Output: an adapted encoder
checkpoint.

### Stage 3 — Supervised finetune (existing pipeline)

Load the CPT'd encoder into Parakeet, finetune on `radiotalk-us-audio-higgs-noisy`
with the tuned HPs. Compare ATCO2 val WER against the SSL-free supervised
baseline to measure CPT's contribution.

## Data flow

TartanAviation (44.1k WAV, unlabeled)
  -> Stage1 prep -> SSL manifest (16k)
  -> Stage2 NEST CPT (encoder init from Parakeet) -> adapted encoder ckpt
  -> Stage3 load encoder into Parakeet + supervised finetune on Higgs
  -> eval on ATCO2 val  (compare vs supervised-only baseline)

## Testing / validation

- Stage 0 spike is the primary correctness gate (weight round-trip).
- Stage 1: assert manifest rows have duration, no text, 16 kHz; spot-check a clip.
- Stage 2: smoke (e.g. 200 steps) before the full overnight CPT; watch SSL loss
  decreasing.
- Stage 3: smoke the encoder-load + 500-step finetune (mirror parakeet-higgs-smoke)
  before the full run.
- Final metric: ATCO2 val WER, CPT+finetune vs finetune-only.

## Open questions / risks

- **Encoder-config alignment (Stage 0)** is the main risk; everything else is
  routine reuse of existing infra.
- 44.1k->16k resample of ~478 hr is a large one-time dump (same GPU-idle issue
  as the supervised dumps; dead `use_lhotse` flag). Acceptable one-time cost.
- CPT step budget is a knob; start modest (continued, not from-scratch) and
  extend if the Stage-3 comparison shows gains.
- Single seed comparisons; treat small WER deltas with the same caution as the
  lr/warmup sweep.

## Out of scope

- Optimizing the WAV-dump path (real Lhotse streaming) — tracked separately.
- Pseudo-labeled self-training (a different semi-supervised axis).

## Run order

End-to-end sequence to reproduce the SSL->finetune pipeline from a fresh clone:

1. **Download TartanAviation** per `docs/data-tartanaviation.md`; set `ssl.source`
   in `configs/train/rtx6kpro/parakeet-ssl-tartan.yaml` to the local glob of WAV
   files (e.g. `data/tartanaviation/**/*.wav`).

2. **SSL continued pretraining** (overnight):
   ```bash
   HF_HUB_OFFLINE=1 rasr train ssl -c configs/train/rtx6kpro/parakeet-ssl-tartan.yaml
   ```
   Output: `ckpt/parakeet-ssl-tartan/final.nemo` — the TartanAviation-adapted
   Parakeet encoder.

3. **Supervised finetune from the CPT'd encoder**:
   ```bash
   HF_HUB_OFFLINE=1 rasr train run -c configs/train/rtx6kpro/parakeet-higgs-ssl.yaml
   ```
   The recipe loads `ckpt/parakeet-ssl-tartan/final.nemo` into the Parakeet
   encoder via `model.init_encoder_from` before finetuning on Higgs-noisy.
   Output: `ckpt/parakeet-higgs-ssl/final.nemo`.

4. **Evaluate and compare WER**:
   ```bash
   rasr eval run -c configs/eval/atco2-val.yaml model=ckpt/parakeet-higgs-ssl/final.nemo
   rasr eval run -c configs/eval/atco2-val.yaml model=ckpt/parakeet-higgs/final.nemo
   ```
   Compare ATCO2 val WER between `parakeet-higgs-ssl` (CPT + finetune) and the
   SSL-free supervised baseline (`parakeet-higgs`) to measure CPT's contribution.
