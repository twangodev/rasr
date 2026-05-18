from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(slots=True, frozen=True)
class AudioInput:
    array: np.ndarray
    sr: int


@runtime_checkable
class Model(Protocol):
    id: str

    def transcribe(self, audios: list[AudioInput]) -> list[str]:
        """Transcribe a batch of audio clips. Implementations decide how to
        schedule the work (GPU batching, concurrent API calls, etc.)."""
        ...
