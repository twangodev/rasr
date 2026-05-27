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

v2 / NOT done here:
  * Line 4 holds ADS-B callsign *codes* (e.g. "EWG7AB"). These are NEVER spoken
    as written — "EWG7AB" is voiced as "Eurowings Seven Alfa Bravo". Biasing the
    raw code is useless; correct handling needs ICAO-operator -> telephony
    expansion plus phonetic-alphabet digit expansion. Deferred to v2.
  * Multiword carrier names ("Air Portugal", "Sky Travel") are emitted as their
    individual words rather than reconstructed phrases. Good enough for biasing
    individual rare tokens; phrase reconstruction is a v2 refinement.
"""

from __future__ import annotations

_MIN_LEN = 3


def _is_biasable(token: str) -> bool:
    """A token is biasable if it is alphabetic and at least _MIN_LEN long.

    This drops pure-numeric tokens, punctuation fragments, and very short
    function words that would only add decoding noise.
    """
    return token.isalpha() and len(token) >= _MIN_LEN


def bias_terms_from_info(info: str) -> list[str]:
    """Parse an ATCO2 `info` field into a de-duplicated list of bias phrases.

    Returns individual alphabetic tokens (len >= 3) drawn from the waypoint
    line and the airline-telephony-names line. Pure-numeric and short tokens
    are dropped. Order is preserved (waypoints first, then airline names).
    """
    if not info:
        return []

    lines = info.split("\n")
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

    seen: set[str] = set()
    out: list[str] = []
    for tok in candidates:
        tok = tok.strip()
        if not _is_biasable(tok):
            continue
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return out
