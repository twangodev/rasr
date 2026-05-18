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

canonical_char_transform = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
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
    return float(
        jiwer.wer(
            ref,
            hyp,
            reference_transform=canonical_transform,
            hypothesis_transform=canonical_transform,
        )
    )


def corpus_cer(refs: list[str], hyps: list[str]) -> float:
    if not refs:
        return float("nan")
    return float(
        jiwer.cer(
            refs,
            hyps,
            reference_transform=canonical_char_transform,
            hypothesis_transform=canonical_char_transform,
        )
    )


def utt_cer(ref: str, hyp: str) -> float:
    return float(
        jiwer.cer(
            ref,
            hyp,
            reference_transform=canonical_char_transform,
            hypothesis_transform=canonical_char_transform,
        )
    )


def summarize(per_utt: list[float], prefix: str) -> dict:
    """Distribution stats (mean/median/p90/max) over per-utterance error rates."""
    if not per_utt:
        return {}
    return {
        f"{prefix}_mean": float(statistics.mean(per_utt)),
        f"{prefix}_median": float(statistics.median(per_utt)),
        f"{prefix}_p90": _percentile(per_utt, 90),
        f"{prefix}_max": float(max(per_utt)),
    }


def _percentile(values: list[float], pct: float) -> float:
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    if f == c:
        return float(sorted_v[f])
    return float(sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f))
