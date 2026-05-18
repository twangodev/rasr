from __future__ import annotations

import numpy as np


class WhisperModel:
    def __init__(self, hf_id: str):
        from transformers import pipeline
        import torch

        self.id = f"whisper:{hf_id}"
        device = 0 if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=hf_id,
            device=device,
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        result = self.pipe({"array": audio, "sampling_rate": sr})
        return result["text"].strip()
