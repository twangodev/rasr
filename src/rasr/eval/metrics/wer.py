from __future__ import annotations

import re
import statistics

import jiwer

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "point", "decimal",
}

# Standard ATC pronunciation variants that radiotalk.normalize doesn't collapse.
_ATC_NUMBER_VARIANTS = {
    "niner": "nine",
    "tree": "three",
    "fife": "five",
}

_DIGIT_RE = re.compile(r"^\d+(?:\.\d+)?$")
_PUNCT_STRIP = ".,!?;:'\""


def _extract_numeric_tokens(text: str) -> str:
    """Return the subsequence of `text` containing only numeric content.

    Keeps cardinal number words (zero–nine, teens, tens, hundred/thousand,
    point/decimal), ATC pronunciation variants (niner→nine, tree→three,
    fife→five), and pure digit substrings. Everything else is dropped.
    """
    tokens = text.lower().split()
    out: list[str] = []
    for tok in tokens:
        clean = tok.strip(_PUNCT_STRIP)
        if not clean:
            continue
        canonical = _ATC_NUMBER_VARIANTS.get(clean, clean)
        if canonical in _NUMBER_WORDS or _DIGIT_RE.match(canonical):
            out.append(canonical)
    return " ".join(out)

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


def corpus_numeric_wer(refs: list[str], hyps: list[str]) -> float:
    """WER computed over only the numeric tokens of each ref/hyp.

    Captures ATC's safety-critical content (callsign digits, headings,
    altitudes, frequencies, runway IDs, squawks). Utterances whose
    reference has no numeric content are excluded from the calculation.
    """
    ref_nums = [_extract_numeric_tokens(r) for r in refs]
    hyp_nums = [_extract_numeric_tokens(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(ref_nums, hyp_nums) if r.strip()]
    if not pairs:
        return float("nan")
    refs_f, hyps_f = zip(*pairs)
    return float(
        jiwer.wer(
            list(refs_f),
            list(hyps_f),
            reference_transform=canonical_transform,
            hypothesis_transform=canonical_transform,
        )
    )


def utt_numeric_wer(ref: str, hyp: str) -> float | None:
    """Per-utterance numeric WER; None if the reference has no numbers."""
    r = _extract_numeric_tokens(ref)
    if not r.strip():
        return None
    h = _extract_numeric_tokens(hyp)
    return float(
        jiwer.wer(
            r,
            h,
            reference_transform=canonical_transform,
            hypothesis_transform=canonical_transform,
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
