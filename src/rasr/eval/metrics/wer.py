from __future__ import annotations

import jiwer


def wer(hyps: list[str], refs: list[str]) -> float:
    return float(jiwer.wer(refs, hyps))
