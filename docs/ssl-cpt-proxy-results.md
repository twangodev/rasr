# SSL continued-pretraining: proxy A/B results

Overnight experiment (2026-05-27) to get a decisive read on whether NEST-style
continued self-supervised pretraining (CPT) of `parakeet-tdt-0.6b-v3`'s encoder
on real ATC audio helps downstream ASR — before committing to a full overnight
CPT + finetune cycle.

## Setup

- **CPT data:** TartanAviation (real US ATC VHF, CMU AirLab). Downloaded a subset
  (~10k recordings / 528 GB available; used a **500-recording subset →
  7,639 VAD-segmented chunks ≈ ~10 hr of speech** for CPT). VAD = energy
  silence-splitting (`librosa.effects.split`, top_db 40), 16 kHz mono.
- **CPT:** NEST (`EncDecDenoiseMaskedTokenPredModel`) initialized from
  Parakeet's encoder, 4000 steps, batch 16, warmup 200. Two variants: naive
  (encoder trains throughout) and **freeze-warmup** (encoder frozen 2000 steps
  while the random NEST decoder warms up, then unfrozen for 2000 steps of gentle
  joint adaptation — `ssl.freeze_encoder_steps`).
- **A/B finetune (proxy):** the tuned short finetune from the lr/warmup sweep —
  3000 steps on 60k Higgs clips, lr 2.5e-5, warmup 300 — with the CPT'd encoder
  loaded vs. not. Metric: best **ATCO2 (real) val WER**.
- **Noise floor:** two identical-config finetune runs gave 0.6153 / 0.6187, so
  run-to-run noise ≈ ±0.003 WER.

## Results

| arm | best ATCO2 val WER | trajectory (step: WER) |
|---|---|---|
| no-CPT control | 0.6187 | 0.635 → 0.632 → 0.621 → 0.621 → **0.619** (smooth, plateau) |
| naive CPT | 0.6641 | 0.833 → 0.703 → **0.664** → 0.672 → 0.667 (bottoms @1640, then rises) |
| **freeze-warmup CPT (seed 1)** | 0.6105 | 0.649 → 0.625 → 0.630 → 0.611 → **0.611** |
| **freeze-warmup CPT (seed 2)** | 0.6202 | 0.649 → 0.626 → **0.620** → 0.626 → 0.635 (rose after 1640) |

Freeze-CPT mean ≈ **0.615** vs no-CPT mean ≈ **0.617** (0.6187 / 0.6153) → a
**tie within the ±0.003 noise floor**. The seed-1 0.6105 was the lucky tail, not
a real effect (seed 2 lands right at baseline).

CPT NEST loss dropped 9.01 (chance floor) → ~5.5–6.4, i.e. the SSL objective is
genuinely learning (not stuck) — but that learning does **not** translate into a
downstream WER gain.

## Findings

1. **Naive NEST CPT actively HURTS** (0.619 → 0.664, +7% relative — far beyond
   the ±0.003 noise floor). NEST attaches a *random* decoder head; early
   gradients training that head perturb the well-aligned pretrained encoder. The
   trajectory confirms damage (bottoms then rises), and it is **not** a
   short-proxy artifact: older 20k-step synthetic-only runs show synthetic-only
   finetune bottoms early (~0.61) then *overfits upward*, so 3k steps already
   captures each encoder's best — naive CPT's floor is simply worse.

2. **Encoder protection neutralizes the harm but gives no gain.** Freeze-warmup
   CPT lands at the no-CPT baseline (mean ~0.615 vs ~0.617 — a tie within noise;
   seed 1 0.6105 / seed 2 0.6202). So protection is necessary to *avoid harm*,
   but the adapted encoder does not actually improve downstream WER on this
   proxy — even though CPT's own SSL loss dropped substantially. Likely the
   protected encoder moved little, and what adaptation it gained didn't help the
   ASR task.

3. **Context that reframes CPT's value:** the synthetic-only baseline (~0.62) is
   far from the production result. Older *mixed* runs (synthetic + real ATCO2/
   ATCOSIM labels folded into finetuning) reach **~0.15 WER**. The dominant lever
   for real-ATC WER is **mixing real labeled data**, not the encoder. So CPT's
   true value must ultimately be judged **on top of the mixed recipe**, not the
   synthetic-only proxy used here.

## Verdict / how to move forward

**Lean NO-GO on SSL CPT (low priority).** Decisive read:

- Naive CPT is **harmful**; freeze-warmup CPT is at best **neutral** (no gain) on
  the proxy. So `freeze_encoder_steps` is mandatory *if* CPT is ever used, but it
  buys nothing here.
- This proxy is where an encoder adapted to real ATC should help **most** — the
  finetune data is purely synthetic, so the model otherwise never sees real
  acoustics. CPT showing zero benefit even here is a fairly strong negative
  signal. On the mixed recipe (where the real-label anchor already exposes the
  model to real ATC), CPT's redundant encoder adaptation is, if anything, *less*
  likely to help.
- **The dominant lever is real labeled data**, not the encoder: synthetic-only
  finetune plateaus ~0.62, while mixing real ATCO2/ATCOSIM labels reaches ~0.15.
  Effort is far better spent on the mixed-data axis (more/better real labeled
  ATC, pseudo-labeling) than on SSL CPT.

**If CPT is revisited anyway,** the one residual test is freeze-warmup CPT on top
of the **mixed** recipe (~20k steps, ≥2 seeds) — but with low expected value
given the above. Possible knobs: more of the ~41 hr available CPT data (used
~10 hr), freeze schedule, lower encoder LR. Recommend deprioritizing.

## Artifacts

- `ssl.freeze_encoder_steps` option: commit `e046418`.
- CPT recipe: `configs/train/rtx6kpro/parakeet-ssl-tartan.yaml`.
- This run's logs under `logs/` (cpt-*, ab-*); checkpoints under `ckpt/` (gitignored).
