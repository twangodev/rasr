from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Model(Protocol):
    id: str

    def transcribe(self, audio: np.ndarray, sr: int) -> str: ...
