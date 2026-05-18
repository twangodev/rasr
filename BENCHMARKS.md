# rasr eval benchmark results

Numbers below come from `rasr eval run`. Each row's underlying artifacts live in `runs/<id>/` (gitignored). Run IDs and dataset commits are recorded in each run's `manifest.json`.

## Conventions

- **Normalization**: hyps and refs are run through `radiotalk.normalize` (callsigns, runways, numbers expanded to spoken form), then a `jiwer` canonical transform (lowercase, strip punctuation) at scoring time.
- **Language hint**: `--language en` unless noted.
- **Whisper generation guards**: `max_new_tokens=256`, `no_repeat_ngram_size=3` (caps the pathological "Thank you. Thank you. ..." failure mode).
- **`n_runaway`**: count of utterances with WER > 1.0 (model generated more wrong words than the reference had).

## ATCO2 validation — real European ATC radio

Dataset: `hf:jlvdoorn/atco2-asr:validation` (N=113). The de facto test set for open ATC ASR research.

| Model | Type | WER | CER | wer_median | cer_median | wer_p90 | n_runaway |
|---|---|---|---|---|---|---|---|
| `jlvdoorn/whisper-large-v3-atco2-asr` | ATC finetune | **0.157** | **0.088** | 0.100 | 0.056 | 0.333 | 0 |
| `jeffreysuu/whisper-atc-finetuned` | ATC finetune (US-EN) | 0.251 | 0.147 | 0.216 | 0.112 | 0.462 | 1 |
| `cohere:CohereLabs/cohere-transcribe-03-2026` | generic | 0.384 | 0.216 | 0.343 | 0.191 | 0.686 | 4 |
| `qwen-asr:Qwen/Qwen3-ASR-1.7B` | generic | 0.398 | 0.233 | 0.364 | 0.211 | 0.709 | 2 |
| `whisper:openai/whisper-large-v3-turbo` | generic | 0.423 | 0.252 | 0.441 | 0.257 | 0.725 | 1 |

**Headline:** the domain finetune beats every generic SOTA model by 2.4× on WER and 2.5× on CER. Note that `jlvdoorn` was trained on the train split of this dataset, so this is in-distribution for it; cross-domain transfer (ATCO2-trained → US ATC) is known to degrade by another ~2×.

## radiotalk-us-audio-tada-noisy — synthetic US ATC

Dataset: `hf:twangodev/radiotalk-us-audio-tada-noisy` (N=100). Synthetic TTS-generated US ATC with VHF degradation pipeline. Useful as a fast-iterate proxy but **optimistic by ~2×** vs real ATCO2.

| Model | Type | WER | wer_median | wer_p90 |
|---|---|---|---|---|
| `jlvdoorn/whisper-large-v3-atco2-asr` | ATC finetune | **0.135** | 0.089 | 0.313 |
| `qwen-asr:Qwen/Qwen3-ASR-1.7B` | generic | 0.180 | 0.106 | 0.451 |
| `jeffreysuu/whisper-atc-finetuned` | ATC finetune (US-EN) | 0.182 | 0.148 | 0.376 |
| `cohere:CohereLabs/cohere-transcribe-03-2026` | generic | 0.197 | 0.154 | 0.500 |
| `whisper:openai/whisper-large-v3-turbo` | generic | 0.205 | 0.162 | 0.429 |
| `parakeet:nvidia/parakeet-tdt-0.6b-v3` | generic | 0.228 | 0.175 | 0.558 |
| `parakeet:nvidia/parakeet-tdt-1.1b` | generic | 0.244 | 0.200 | 0.613 |
| `granite-speech:ibm-granite/granite-speech-4.1-2b` | generic | 0.332 | 0.270 | 0.791 |

(CER not yet recorded for this set — runs predate the metric. Re-run any row to backfill.)

## Cross-dataset gap

Same models, two datasets:

| Model | radiotalk WER | ATCO2 WER | ratio |
|---|---|---|---|
| `jlvdoorn` (trained on ATCO2) | 0.135 | 0.157 | 1.16× |
| `jeffreysuu` (trained on US ATC) | 0.182 | 0.251 | 1.38× |
| `cohere` | 0.197 | 0.384 | 1.95× |
| `qwen3-asr-1.7b` | 0.180 | 0.398 | 2.21× |
| `whisper-large-v3-turbo` | 0.205 | 0.423 | 2.06× |

Real ATC radio is meaningfully harder than synthetic noisy TTS, especially for generic models. ATC-trained models generalize better — synthetic-vs-real gap shrinks with prior domain exposure.

## Caveats

- **Statistical power**: N=113 on ATCO2 ≈ 2,200 reference words → WER 95% CI ≈ ±2pp; CER 95% CI ≈ ±0.5pp (chars are ~5× more numerous than words). Differences smaller than the CI are not significant.
- **Within the generic tier** (Cohere/Qwen/Whisper-turbo on ATCO2), CER separations are significant; WER ones are borderline.
- **No bootstrap CIs yet** — point estimates only. Adding paired bootstrap would tighten the comparisons further.
- **jlvdoorn caveat**: the ATCO2 validation split overlaps its training distribution. Reported 13.5% on their paper, we measure 15.7% (small delta from normalization differences).
- **Normalization gaps**: `radiotalk.normalize` doesn't expand decimals (`125.7` stays unexpanded), uniformly inflating WER for all models.

## Reproducing

```bash
# ATCO2 validation, jlvdoorn finetune
rasr eval run \
  -m whisper:jlvdoorn/whisper-large-v3-atco2-asr \
  -d hf:jlvdoorn/atco2-asr:validation \
  --language en

# Generic baseline
rasr eval run \
  -m qwen-asr:Qwen/Qwen3-ASR-1.7B \
  -d hf:jlvdoorn/atco2-asr:validation \
  --language en --batch-size 16
```

Each run writes `runs/<timestamp>-<hash>/{manifest.json, transcripts.jsonl, scores.json}`. Manifests pin the model id, dataset id, git SHA, and `radiotalk.normalize` version so numbers are attributable later.
