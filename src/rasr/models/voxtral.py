from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from rasr.models.base import AudioInput

TARGET_SR = 16000


class VoxtralModel:
    """Mistral Voxtral adapter (audio-conditioned LLM transcription).

    Voxtral is an LLM-decoder ASR: a Whisper-style audio encoder feeds into a
    Ministral text decoder. Transcription uses the high-level helper
    `processor.apply_transcription_request(...)`, which builds the chat-templated
    prompt + audio inputs for us. The helper accepts file paths for `audio`, so
    we write each clip to a temp WAV (per-clip loop — batching across utterances
    is supported by the processor but requires padding handling we don't need
    for an eval-only adapter).
    """

    def __init__(
        self,
        hf_id: str,
        language: str | None = None,
        max_new_tokens: int = 500,
    ):
        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        self.id = f"voxtral:{hf_id}"
        self.hf_id = hf_id
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            hf_id,
            torch_dtype=self._dtype,
            device_map=self._device,
        )
        self.model.eval()
        self.language = language or "en"
        self.max_new_tokens = max_new_tokens
        self._tmp_dir = tempfile.mkdtemp(prefix="rasr-voxtral-")

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == TARGET_SR:
            return audio.astype(np.float32)
        import librosa

        return librosa.resample(
            audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR
        )

    def transcribe(self, audios: list[AudioInput]) -> list[str]:
        if not audios:
            return []
        import soundfile as sf
        import torch

        results: list[str] = []
        for i, a in enumerate(audios):
            arr = self._resample(a.array, a.sr)
            wav_path = Path(self._tmp_dir) / f"clip_{i:08d}.wav"
            sf.write(str(wav_path), arr, TARGET_SR, subtype="PCM_16")

            inputs = self.processor.apply_transcription_request(
                language=self.language,
                audio=str(wav_path),
                model_id=self.hf_id,
            )
            inputs = inputs.to(self._device, dtype=self._dtype)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
            decoded = self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            results.append((decoded[0] if decoded else "").strip())

            try:
                wav_path.unlink()
            except OSError:
                pass

        return results
