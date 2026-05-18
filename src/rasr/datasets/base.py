from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np


@dataclass(slots=True)
class Utterance:
    utt_id: str
    audio: np.ndarray
    sr: int
    reference: str
    meta: dict = field(default_factory=dict)


class Dataset(Protocol):
    id: str

    def __iter__(self) -> Iterator[Utterance]: ...
