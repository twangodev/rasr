# Higgs finetuning lr/warmup sweep

Tuning the **supervised finetuning** hyperparameters for
`nvidia/parakeet-tdt-0.6b-v3` on the new synthetic corpus
`twangodev/radiotalk-us-audio-higgs-noisy` (Higgs-Audio TTS, 8 kHz source,
566k clips). Run 2026-05-27.

## Method

Short proxy runs over a shared, cached 60k-clip WAV dump (see
`configs/train/rtx6kpro/parakeet-higgs-sweep.yaml` and
`scripts/sweep_higgs.sh`):

- 3000 steps/trial, batch 48, bf16-mixed, ATCO2 `validation` as the metric set.
- Selection metric: best ATCO2 val WER over the trial.
- Two stages, because **short proxy runs cannot fairly compare warmup**: NeMo's
  CosineAnnealing ties its decay horizon to `max_steps`, so a warmup near
  `max_steps` is unfairly truncated. We therefore swept LR at a fixed small
  warmup, then probed warmup (only small fractions of the horizon) at the
  winning LR.

Note: the val set is **real** ATCO2 audio, so WER measures transfer from the
synthetic Higgs distribution — not memorization of TTS artifacts. Absolute WER
is high (~0.62) because this is synthetic-only, no real-ATC anchor, only 3k
steps — these numbers are for *relative* comparison, not a quality bar
(jlvdoorn's anchored reference is ~0.157).

## Results

Stage A — learning rate (warmup=300):

| lr      | val WER |
|---------|---------|
| 2.5e-5  | **0.6153** |
| 5e-5    | 0.6240 |
| 7.5e-5  | 0.6245 |
| 1e-4    | 0.6211 |
| 1.5e-4  | 0.6279 |
| 2e-4    | skipped (high-LR, non-informative) |

Stage B — warmup (lr=2.5e-5):

| warmup | val WER |
|--------|---------|
| 100    | 0.6202 |
| 300    | **0.6153** |
| 600    | 0.6202 |

Best overall: **lr=2.5e-5, warmup=300 → 0.6153**.

## Takeaways

Both knobs are weak levers at this horizon — the full spread is ~2% relative —
so the conclusions are directional:

1. **Low LR wins.** 2.5e-5 is best; LR ≥ 1.5e-4 clearly degrades. Consistent
   with avoiding catastrophic forgetting when adapting a strong pretrained
   model within the same language/tokenizer.
2. **Short warmup wins.** 100–600 are all fine (shallow U, min at ~300). The
   inherited default of **5000 was badly miscalibrated** for these short
   finetuning schedules — this was the main finding.
3. Differences inside the low-LR / short-warmup region are near single-seed
   noise; don't over-fit to the exact 2.5e-5 / 300.

## Recommended supervised-finetuning HPs

- `optimizer.lr: 2.5e-5`
- `scheduler.warmup_steps: 500` (the winner was 300; rounding up slightly is
  still well within the "short" regime and a touch safer for a longer
  20k–50k-step run)
- Everything else from `base.yaml` unchanged.

These apply to the supervised finetune whether or not it follows SSL
pretraining of the encoder (the planned final-run direction).

## Caveats / possible follow-ups

- Single seed per trial; the intra-region spread is within plausible noise.
- 3k-step proxy; a mid LR could partially catch up at the real 20k+ horizon.
  Low LR remains the safe choice for not damaging the base model.
- The LR floor (2.5e-5) was the grid minimum and the winner — the true optimum
  may sit slightly lower (e.g. 1e-5). Worth one extra probe only if a full
  baseline at 2.5e-5 underwhelms.
