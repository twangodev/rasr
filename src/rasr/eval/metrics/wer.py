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

# Single-word number tokens -> their digit/symbol form. Used by the
# digit-aware WER so that "two one zero" and "210" compare as equal. Only the
# unambiguous single-digit cardinals (plus ATC variants and the decimal point)
# are mapped; multi-word composites like "two hundred" are intentionally left
# alone (collapsing them reliably needs a full number parser, which is out of
# scope — ATC speech overwhelmingly uses digit-by-digit readout anyway).
_WORD_TO_DIGIT = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    # ATC pronunciation variants.
    "niner": "9",
    "tree": "3",
    "fife": "5",
    # Decimal markers (frequencies, "one one eight decimal one").
    "point": ".",
    "decimal": ".",
}


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


def _map_number_words(text: str) -> str:
    """Replace single-word number tokens with their digit/symbol form.

    Operates token-by-token on whitespace-split text. A trailing/leading
    decimal point produced from "point"/"decimal" is left as a standalone
    "." token; downstream RemovePunctuation in the canonical transform handles
    any residue, but the digit tokens themselves survive so "two one zero"
    becomes "2 1 0" and matches "210" only after the contiguous-digit merge
    below. To make "two one zero" == "210" we also glue runs of adjacent
    single-digit tokens together.
    """
    raw = text.lower().split()
    mapped: list[str] = []
    for tok in raw:
        clean = tok.strip(_PUNCT_STRIP)
        mapped.append(_WORD_TO_DIGIT.get(clean, tok))
    # Merge runs of adjacent single digits into one number token so that a
    # digit-by-digit readout ("2 1 0") collapses to the written form ("210").
    out: list[str] = []
    buf: list[str] = []
    for tok in mapped:
        if len(tok) == 1 and tok.isdigit():
            buf.append(tok)
        elif tok == ".":
            buf.append(".")
        else:
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(tok)
    if buf:
        out.append("".join(buf))
    return " ".join(out)


class _NumberWordsToDigits:
    """jiwer-compatible transform: number-word normalization, then canonical."""

    def __call__(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        return canonical_transform([_map_number_words(s) for s in sentences])


digit_aware_transform = _NumberWordsToDigits()


def wer_digit_aware(refs: list[str], hyps: list[str]) -> float:
    """Corpus WER where spoken digits and written digits compare as equal.

    Number-words ("two one zero", with ATC variants niner/tree/fife and
    "point"/"decimal") are mapped to digit tokens and adjacent single digits
    are merged ("2 1 0" -> "210") on BOTH references and hypotheses before the
    standard canonical (lowercase/strip-punct) WER is computed. This removes
    the digit-spelling mismatch that otherwise inflates ATC WER and adds noise
    to biased-vs-unbiased comparisons.
    """
    if not refs:
        return float("nan")
    return float(
        jiwer.wer(
            refs,
            hyps,
            reference_transform=digit_aware_transform,
            hypothesis_transform=digit_aware_transform,
        )
    )


def utt_wer_digit_aware(ref: str, hyp: str) -> float:
    """Per-utterance digit-aware WER (see :func:`wer_digit_aware`)."""
    return float(
        jiwer.wer(
            ref,
            hyp,
            reference_transform=digit_aware_transform,
            hypothesis_transform=digit_aware_transform,
        )
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


# Inline sanity checks for the digit-aware WER (cheap; runs at import only when
# this module is executed directly).
if __name__ == "__main__":
    assert wer_digit_aware(["turn two one zero"], ["turn 210"]) == 0.0, (
        "digit-aware WER should treat 'two one zero' == '210'"
    )
    assert wer_digit_aware(["niner tree fife"], ["935"]) == 0.0, (
        "ATC variants niner/tree/fife should map to 9/3/5 and merge"
    )
    assert wer_digit_aware(["one one eight decimal one"], ["118.1"]) == 0.0, (
        "'decimal' should map to '.' and merge into the frequency"
    )
    # A genuine digit error must still register.
    assert wer_digit_aware(["turn two one zero"], ["turn 220"]) > 0.0
    print("wer_digit_aware sanity checks passed")
