from __future__ import annotations

from rasr.models.base import AudioInput


class WhisperModel:
    def __init__(
        self,
        hf_id: str,
        max_new_tokens: int = 256,
        no_repeat_ngram_size: int = 3,
        language: str | None = None,
    ):
        from transformers import pipeline
        import torch

        self.id = f"whisper:{hf_id}"
        device = 0 if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=hf_id,
            device=device,
            torch_dtype=dtype,
        )
        self._generate_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        if language is not None:
            self._generate_kwargs["language"] = language
            self._generate_kwargs["task"] = "transcribe"

    def transcribe(self, audios: list[AudioInput]) -> list[str]:
        if not audios:
            return []
        inputs = [{"array": a.array, "sampling_rate": a.sr} for a in audios]
        results = self.pipe(
            inputs,
            generate_kwargs=self._generate_kwargs,
            batch_size=len(inputs),
        )
        return [r["text"].strip() for r in results]
