"""Spike: can NeMo TDT confidence gate pseudo-labels for TartanAviation?

Goal / make-or-break question
-----------------------------
We want to pseudo-label TartanAviation (real, UNLABELED US ATC) with a finetuned
TDT model and add only the *trustworthy* transcripts to training. This spike
checks the load-bearing assumption: does NeMo's per-utterance TDT confidence
RANK-ORDER transcript quality? If high-confidence == clean ATC and
low-confidence == garbage (empty / repetitive / gibberish / single word), we can
filter on a threshold. If confidence is flat or anti-correlated with quality,
confidence-only filtering is dead and we need a fallback (two-model agreement,
LM scoring).

There is NO ground truth for these clips, so this is a QUALITATIVE correlation
check (confidence vs eyeball quality), not a WER measurement. That is expected.

Pinned NeMo 2.7 API (verified by inspecting a Hypothesis object)
----------------------------------------------------------------
  - Model: ckpt/parakeet-mixed-bandpass/final.nemo is an EncDecRNNTBPEModel
    (TDT transducer + aux CTC). Restored with `.restore_from(..., map_location="cpu")`.
  - Enable confidence by copying model.cfg.decoding, setting strategy
    "greedy_batch" and a `confidence_cfg` block (preserve_word/token/frame
    confidence; entropy/tsallis method), then `model.change_decoding_strategy(cfg)`.
  - `model.transcribe(paths, return_hypotheses=True)` then yields `Hypothesis`
    objects whose populated confidence fields are:
       * h.word_confidence  -> list[Tensor], one scalar in [0,1] per WORD
       * h.token_confidence -> list[Tensor], one scalar per emitted TOKEN
       * h.frame_confidence -> list[Tensor], one scalar per FRAME
       * h.score            -> raw transducer log-prob sum (NOT length-normalized;
                               strongly anti-correlated with length, so NOT a good
                               quality gate on its own).
    These are per-step confidences in [0,1] (1.0 == fully confident). We reduce
    word_confidence to a per-utterance scalar. We report mean AND min word
    confidence: min is the more useful quality gate because a single low-confidence
    word (a hallucinated/garbled token) is exactly what we want to catch.

CPU ONLY. Run with `CUDA_VISIBLE_DEVICES=""`.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/spikes/pseudo_label_confidence_spike.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

# Belt-and-suspenders: force CPU even if the caller forgot the env var.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import nemo.collections.asr as nemo_asr  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

MODEL_PATH = "ckpt/parakeet-mixed-bandpass/final.nemo"
# Largest cached SSL manifest = the ~7k TartanAviation VAD chunks.
SSL_MANIFEST = "data/cache/ssl/ssl__local__1e9bdc537bd9/manifest.jsonl"
NUM_CLIPS = 20
# Stride through the manifest so we don't just sample one recording session.
STRIDE = 53


def build_confidence_decoding_cfg(base_decoding_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(base_decoding_cfg, resolve=True))
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        cfg.confidence_cfg = {
            "preserve_frame_confidence": True,
            "preserve_token_confidence": True,
            "preserve_word_confidence": True,
            "aggregation": "mean",
            "method_cfg": {
                "name": "entropy",
                "entropy_type": "tsallis",
                "alpha": 0.33,
                "entropy_norm": "lin",
            },
        }
    return cfg


def to_floats(conf_list) -> list[float]:
    """word/token/frame_confidence are lists of 0-dim tensors; flatten to floats."""
    if conf_list is None:
        return []
    out = []
    for c in conf_list:
        try:
            out.append(float(c))
        except (TypeError, ValueError):
            pass
    return out


def quality_flag(text: str) -> str:
    """Heuristic eyeball label for a transcript (no ground truth available).

    Returns one of: EMPTY, SINGLE-WORD, REPETITIVE, OK. This is a coarse junk
    detector, NOT a WER proxy -- it just helps us see whether confidence tracks
    the obvious-garbage cases.
    """
    t = text.strip()
    if not t:
        return "EMPTY"
    words = t.split()
    if len(words) <= 1:
        return "SINGLE-WORD"
    # Repetition: same token dominates, or an n-gram repeats a lot.
    counts = Counter(w.lower() for w in words)
    top_word, top_n = counts.most_common(1)[0]
    if top_n >= max(3, len(words) * 0.5):
        return "REPETITIVE"
    # Immediate token repeats ("the the the").
    repeats = sum(1 for a, b in zip(words, words[1:]) if a.lower() == b.lower())
    if repeats >= 3:
        return "REPETITIVE"
    return "OK"


def main() -> int:
    print(f"[load] {MODEL_PATH} on CPU ...", flush=True)
    model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location="cpu")
    model = model.cpu()
    model.eval()
    print(f"[load] type={type(model).__name__}", flush=True)

    cfg = build_confidence_decoding_cfg(model.cfg.decoding)
    model.change_decoding_strategy(cfg)

    # Sample chunks spread across the manifest.
    paths: list[str] = []
    with open(SSL_MANIFEST) as f:
        for i, line in enumerate(f):
            if i % STRIDE == 0:
                paths.append(json.loads(line)["audio_filepath"])
            if len(paths) >= NUM_CLIPS:
                break
    print(f"[data] {len(paths)} TartanAviation chunks from {SSL_MANIFEST}", flush=True)

    print("[decode] transcribing with confidence ...", flush=True)
    hyps = model.transcribe(paths, batch_size=4, return_hypotheses=True, verbose=False)

    rows = []
    for path, h in zip(paths, hyps):
        text = (h.text or "").strip()
        wc = to_floats(h.word_confidence)
        tc = to_floats(h.token_confidence)
        mean_wc = sum(wc) / len(wc) if wc else 0.0
        min_wc = min(wc) if wc else 0.0
        mean_tc = sum(tc) / len(tc) if tc else 0.0
        rows.append(
            {
                "name": os.path.basename(path),
                "text": text,
                "mean_word_conf": mean_wc,
                "min_word_conf": min_wc,
                "mean_token_conf": mean_tc,
                "score": float(h.score) if h.score is not None else float("nan"),
                "flag": quality_flag(text),
            }
        )

    # ---- Report, sorted by MIN word confidence (the proposed gate) ----
    rows.sort(key=lambda r: r["min_word_conf"], reverse=True)

    print("\n" + "=" * 100)
    print("PER-UTTERANCE CONFIDENCE (sorted by MIN word confidence, high -> low)")
    print("  columns: min_wc / mean_wc / mean_token_c / raw_score / heuristic_flag")
    print("=" * 100)
    for r in rows:
        print(
            f"\nmin_wc={r['min_word_conf']:.3f}  mean_wc={r['mean_word_conf']:.3f}  "
            f"mean_tc={r['mean_token_conf']:.3f}  score={r['score']:8.1f}  "
            f"[{r['flag']:11s}]  {r['name']}"
        )
        print(f"    TEXT: {r['text'] if r['text'] else '<empty>'}")

    # ---- Top / bottom buckets ----
    print("\n" + "=" * 100)
    print("HIGHEST-CONFIDENCE (top 5 by min_word_conf) -- should look like clean ATC")
    print("=" * 100)
    for r in rows[:5]:
        print(f"  min_wc={r['min_word_conf']:.3f} [{r['flag']:11s}] {r['text'] or '<empty>'}")

    print("\n" + "=" * 100)
    print("LOWEST-CONFIDENCE (bottom 5 by min_word_conf) -- should look like junk")
    print("=" * 100)
    for r in rows[-5:]:
        print(f"  min_wc={r['min_word_conf']:.3f} [{r['flag']:11s}] {r['text'] or '<empty>'}")

    # ---- Quick separation summary ----
    ok = [r for r in rows if r["flag"] == "OK"]
    junk = [r for r in rows if r["flag"] != "OK"]
    print("\n" + "=" * 100)
    print("SEPARATION (heuristic OK vs junk; eyeball, not WER)")
    print("=" * 100)

    def stat(name, key, group):
        if not group:
            print(f"  {name:16s} {key}: n=0")
            return
        vals = [r[key] for r in group]
        print(
            f"  {name:16s} {key}: n={len(group)} "
            f"min={min(vals):.3f} mean={sum(vals)/len(vals):.3f} max={max(vals):.3f}"
        )

    stat("OK transcripts", "min_word_conf", ok)
    stat("junk transcripts", "min_word_conf", junk)
    stat("OK transcripts", "mean_word_conf", ok)
    stat("junk transcripts", "mean_word_conf", junk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
