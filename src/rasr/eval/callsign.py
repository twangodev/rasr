"""Expand ATCO2 ADS-B callsign *codes* into their spoken (voiced) forms.

The ATCO2 `info` block carries, on its ADS-B line, the callsign codes of the
aircraft that were on frequency, e.g. ``EWG7AB PGT302 OKLBA WZZ9654``. These
codes are NEVER spoken as written. Controllers/pilots voice them in one of two
ways:

  * AIRLINE callsign: a 3-letter ICAO *operator* code + a flight identifier.
    The operator code is spoken as the carrier's telephony name and the
    identifier is read out character-by-character (NATO phonetic for letters,
    number words for digits)::

        EWG7AB   -> "eurowings seven alfa bravo"
        RYR73AH  -> "ryanair seven three alfa hotel"
        WZZ9654  -> "wizz air nine six five four"

  * GA / REGISTRATION: an aircraft registration (or an unknown operator). The
    whole code is spelled out phonetically::

        OKLBA    -> "oscar kilo lima bravo alfa"
        OEGWV    -> "oscar echo golf whiskey victor"

This module provides :func:`expand_callsign`, which turns a single code into
its spoken phrase, returning ``None`` when it cannot sensibly expand.

Casing: everything is returned lowercased. The downstream biasing / WER are
case-insensitive (the boosting tree and the digit-aware WER both lowercase).
"""

from __future__ import annotations

import re

# ICAO 3-letter operator code -> spoken telephony name (lowercased).
#
# Only mappings we are confident about are included. A WRONG telephony name
# biases the decoder toward a phrase that never occurs, which can only hurt, so
# when in doubt we OMIT the code (it then falls through to phonetic spelling,
# which is at worst neutral). Forms are chosen to match how the name actually
# appears in the ATCO2 transcripts where observed (e.g. "wizz air", "sunturk").
ICAO_TELEPHONY: dict[str, str] = {
    "EWG": "eurowings",
    "RYR": "ryanair",
    "DLH": "lufthansa",
    "QTR": "qatari",
    "WZZ": "wizz air",
    "CSA": "csa",
    "TAP": "air portugal",
    "UAE": "emirates",
    "AFR": "airfrans",
    "BAW": "speedbird",
    "EZS": "easy",
    "EZY": "easy",
    "PGT": "sunturk",
    "VLG": "vueling",
    "SWR": "swiss",
    "AUA": "austrian",
    "TRA": "transavia",
    "LOT": "lot",
    "SDR": "sundair",
    "TVS": "sky travel",
    "BLA": "dark blue",
    "IRA": "iranair",
    "ROT": "tarom",
    "VOE": "volotea",
    "ELY": "elal",
    # Additional carriers observed in the European ATCO2 transcripts.
    "KLM": "klm",
    "THY": "turkish",
    "AZA": "alitalia",
    "ACA": "air canada",
    "UAL": "united",
    "FIN": "finnair",
    "AAL": "american",
    "BTI": "air baltic",
    "SAS": "scandinavian",
    "AEE": "aegean",
    "TUI": "tuifly",
    "SXS": "sunexpress",
    "EJU": "alpine",
    "RAM": "royalair maroc",
    "AHO": "air hamburg",
    "ETD": "etihad",
}

# NATO phonetic alphabet (lowercased). "alfa"/"juliett" are the ICAO
# spellings; both "alfa"/"alpha" occur in the transcripts but we standardise on
# the ICAO form.
_NATO: dict[str, str] = {
    "A": "alfa",
    "B": "bravo",
    "C": "charlie",
    "D": "delta",
    "E": "echo",
    "F": "foxtrot",
    "G": "golf",
    "H": "hotel",
    "I": "india",
    "J": "juliett",
    "K": "kilo",
    "L": "lima",
    "M": "mike",
    "N": "november",
    "O": "oscar",
    "P": "papa",
    "Q": "quebec",
    "R": "romeo",
    "S": "sierra",
    "T": "tango",
    "U": "uniform",
    "V": "victor",
    "W": "whiskey",
    "X": "x-ray",
    "Y": "yankee",
    "Z": "zulu",
}

# Digits as spoken. The transcripts use both "nine" and "niner"; "nine" is the
# more common form, so we emit that (the digit-aware WER would collapse them
# anyway, but the boosting tree matches on the literal phrase).
_DIGIT: dict[str, str] = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

_CODE_RE = re.compile(r"^[A-Z0-9]+$")


def _spoken(chars: str) -> str | None:
    """Spell out a run of letters/digits character-by-character.

    Returns the space-joined spoken tokens, or ``None`` if any character is
    neither a letter nor a digit.
    """
    out: list[str] = []
    for ch in chars:
        if ch in _NATO:
            out.append(_NATO[ch])
        elif ch in _DIGIT:
            out.append(_DIGIT[ch])
        else:
            return None
    return " ".join(out) if out else None


def expand_callsign(code: str) -> str | None:
    """Expand a single ADS-B callsign code into its spoken form (lowercased).

    Strategy:
      * Uppercase/clean the code. Reject anything with non-alnum characters.
      * If the leading alphabetic prefix (1-3 letters) is a known ICAO operator
        in :data:`ICAO_TELEPHONY` AND there is a remainder after it, emit
        ``telephony + spoken(remainder)``.
      * Otherwise treat it as a registration / unknown operator and spell the
        whole code phonetically.
      * Return ``None`` if it cannot be expanded (empty / bad characters).
    """
    if not code:
        return None
    code = code.strip().upper()
    if not code or not _CODE_RE.match(code):
        return None

    # Leading alphabetic prefix (the operator-code candidate).
    m = re.match(r"^([A-Z]{1,3})(.*)$", code)
    if m:
        prefix, rest = m.group(1), m.group(2)
        if prefix in ICAO_TELEPHONY and rest:
            spoken_rest = _spoken(rest)
            if spoken_rest is not None:
                return f"{ICAO_TELEPHONY[prefix]} {spoken_rest}"

    # Registration / unknown operator: spell the whole thing.
    return _spoken(code)


# Inline sanity checks (cheap; run only when executed directly).
if __name__ == "__main__":
    assert expand_callsign("EWG7AB") == "eurowings seven alfa bravo", expand_callsign("EWG7AB")
    assert expand_callsign("OKLBA") == "oscar kilo lima bravo alfa", expand_callsign("OKLBA")
    assert expand_callsign("RYR73AH") == "ryanair seven three alfa hotel", expand_callsign("RYR73AH")
    wzz = expand_callsign("WZZ9654")
    assert "wizz air" in wzz and "nine six five four" in wzz, wzz
    assert expand_callsign("PGT302") == "sunturk three zero two", expand_callsign("PGT302")
    assert expand_callsign("OEGWV") == "oscar echo golf whiskey victor", expand_callsign("OEGWV")
    assert expand_callsign("") is None
    assert expand_callsign("OK-LBA") is None  # bad char -> None
    print("expand_callsign sanity checks passed")
    # Show a few expansions for eyeballing.
    for c in ["EWG7AB", "OKLBA", "WZZ9654", "TVS432P", "OKPRM", "K01"]:
        print(f"  {c:10s} -> {expand_callsign(c)}")
