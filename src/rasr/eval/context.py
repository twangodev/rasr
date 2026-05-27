"""Parse ATCO2 `info` blocks into decode-time contextual-biasing terms.

The ATCO2 dataset attaches a per-clip `info` field that is newline-separated:

    line 0: airport ICAO code        (e.g. "LKPR")        -- skipped
    line 1: airport name             (e.g. "Praha Ruzyne") -- skipped (v2)
    line 2: sector / facility        (e.g. "Radar")        -- skipped
    line 3: WAYPOINTS                (e.g. "AKEVA ARVEG …") -- spoken verbatim
    line 4: ADS-B callsign codes     (e.g. "EWG7AB PGT302") -- NOT spoken verbatim
    line 5+: AIRLINE TELEPHONY NAMES (e.g. "Eurowings Wizzair …") -- spoken

This module turns that into a flat list of bias phrases suitable for NeMo's
transducer boosting tree.

We use TWO sources:

  * The WAYPOINTS line (line index 3): pilots/controllers read these fixes
    verbatim ("cleared direct ARVEG"), so each token is a high-value bias
    phrase.
  * The LAST non-empty line (airline telephony names): the spoken form of a
    carrier ("Eurowings", "Wizzair", "Qatari"), each word is a bias phrase.

v2 (mode="v2", now the default in scripts/eval_biased.py):
  * Line 4 (ADS-B callsign codes) IS now used: each code is expanded into its
    spoken phrase via :func:`rasr.eval.callsign.expand_callsign`
    ("EWG7AB" -> "eurowings seven alfa bravo", "OKLBA" -> "oscar kilo lima
    bravo alfa"). Codes that cannot be expanded are dropped.
  * The waypoints line (index 3) is kept — these are precise per-clip nav fixes.
  * The generic line-5 airline-telephony word shotgun is DROPPED. In v1 it dumped
    ~20 carrier names + waypoints for every clip and over-biased to a net-neutral
    result. In v2 the carrier names instead arrive *inside* the expanded
    callsigns, scoped to the aircraft actually on frequency.
"""

from __future__ import annotations

from rasr.eval.callsign import expand_callsign

_MIN_LEN = 3


def _is_biasable(token: str) -> bool:
    """A token is biasable if it is alphabetic and at least _MIN_LEN long.

    This drops pure-numeric tokens, punctuation fragments, and very short
    function words that would only add decoding noise.
    """
    return token.isalpha() and len(token) >= _MIN_LEN


def _dedup_preserve(items: list[str]) -> list[str]:
    """De-duplicate (case-insensitively) while preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        it = it.strip()
        if not it:
            continue
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _bias_terms_v1(lines: list[str]) -> list[str]:
    """v1: individual alphabetic tokens (len >= 3) from the waypoint line and
    the airline-telephony-names line. Pure-numeric / short tokens dropped."""
    candidates: list[str] = []

    # Waypoints: line index 3 (the 4th line). Spoken verbatim.
    if len(lines) > 3 and lines[3].strip():
        candidates.extend(lines[3].split())

    # Airline telephony names: the LAST non-empty line. Spoken form.
    # Find it by scanning from the end; guard against it being the waypoint
    # line itself (clips with no airline line, e.g. only 4 populated lines).
    last_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_idx = i
            break
    if last_idx > 3:
        # Slashes separate alternate names ("Csa / Czech"); split on them too.
        candidates.extend(lines[last_idx].replace("/", " ").split())

    return _dedup_preserve([t for t in candidates if _is_biasable(t)])


def _bias_terms_v2(lines: list[str]) -> list[str]:
    """v2: waypoints (line 3) PLUS expanded ADS-B callsigns (line 4).

    Each callsign code on the ADS-B line is expanded into its spoken phrase
    (e.g. "EWG7AB" -> "eurowings seven alfa bravo") via
    :func:`rasr.eval.callsign.expand_callsign`; codes that can't expand are
    dropped. The generic line-5 airline-name shotgun is intentionally omitted
    (the carrier names now arrive inside the expanded callsigns).
    """
    candidates: list[str] = []

    # ADS-B callsign codes: line index 4. Expand each to its spoken phrase.
    # These are listed FIRST: they are the highest-value, per-clip-specific bias
    # phrases, so if a downstream cap truncates the list it drops waypoints (a
    # near-constant airport-wide set) rather than callsigns.
    if len(lines) > 4 and lines[4].strip():
        for code in lines[4].split():
            phrase = expand_callsign(code)
            if phrase:
                candidates.append(phrase)

    # Waypoints: line index 3. Spoken verbatim; keep biasable tokens.
    if len(lines) > 3 and lines[3].strip():
        candidates.extend(t for t in lines[3].split() if _is_biasable(t))

    return _dedup_preserve(candidates)


def bias_terms_from_info(info: str, mode: str = "v2") -> list[str]:
    """Parse an ATCO2 `info` field into a de-duplicated list of bias phrases.

    mode="v2" (default): waypoints (line 3) + spoken-expanded ADS-B callsigns
    (line 4). mode="v1": individual alphabetic tokens from the waypoint line and
    the airline-telephony-names line (the original over-biasing behaviour, kept
    reachable for comparison). Order is preserved (waypoints first).
    """
    if not info:
        return []
    lines = info.split("\n")
    if mode == "v1":
        return _bias_terms_v1(lines)
    if mode == "v2":
        return _bias_terms_v2(lines)
    raise ValueError(f"unknown bias mode: {mode!r} (expected 'v1' or 'v2')")
