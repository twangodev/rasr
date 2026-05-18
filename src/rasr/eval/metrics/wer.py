from __future__ import annotations

import statistics

import jiwer

canonical_transform = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """Aggregate WER across all utterances (jiwer's default — weighted by ref length)."""
    if not refs:
        return float("nan")
    return float(
        jiwer.wer(
            refs,
            hyps,
            reference_transform=canonical_transform,
            hypothesis_transform=canonical_transform,
        )
    )


def utt_wer(ref: str, hyp: str) -> float:
    """WER for a single utterance."""
    return float(
        jiwer.wer(
            ref,
            hyp,
            reference_transform=canonical_transform,
            hypothesis_transform=canonical_transform,
        )
    )


def summarize(per_utt: list[float]) -> dict:
    """Distribution stats over per-utterance WER values.

    Surfaces median/p90/runaway-count so a few hallucinations don't silently
    dominate the headline corpus WER.
    """
    if not per_utt:
        return {"n": 0}
    return {
        "n": len(per_utt),
        "wer_mean": float(statistics.mean(per_utt)),
        "wer_median": float(statistics.median(per_utt)),
        "wer_p90": _percentile(per_utt, 90),
        "wer_max": float(max(per_utt)),
        "n_runaway": sum(1 for w in per_utt if w > 1.0),
    }


def _percentile(values: list[float], pct: float) -> float:
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    if f == c:
        return float(sorted_v[f])
    return float(sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f))
