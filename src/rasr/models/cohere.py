from __future__ import annotations

import numpy as np

from rasr.models.base import AudioInput

TARGET_SR = 16000


class CohereAsrModel:
    """Cohere Transcribe adapter (open-weight Conformer ASR).

    Per-clip inference loop; the upstream processor chunks audio >35s
    internally and re-assembles via `audio_chunk_index`, so we leave that
    handling to it rather than batching across utterances.
    """

    def __init__(
        self,
        hf_id: str,
        language: str = "en",
        punctuation: bool = True,
        max_new_tokens: int = 440,
    ):
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        self.id = f"cohere:{hf_id}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            hf_id,
            device_map=device,
            dtype=dtype,
        )
        self.model.eval()
        self.language = language
        self.punctuation = punctuation
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._dtype = dtype

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == TARGET_SR:
            return audio
        import librosa

        return librosa.resample(
            audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR
        )

    def transcribe(self, audios: list[AudioInput]) -> list[str]:
        if not audios:
            return []
        import torch

        results: list[str] = []
        for a in audios:
            audio_16k = self._resample(a.array, a.sr)
            inputs = self.processor(
                audio=audio_16k,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                language=self.language,
                punctuation=self.punctuation,
            )
            audio_chunk_index = inputs.get("audio_chunk_index")
            inputs = inputs.to(self._device, dtype=self._dtype)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens
                )
            decoded = self.processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=self.language,
            )
            text = decoded[0] if isinstance(decoded, list) else decoded
            results.append(text.strip())
        return results
