from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from rasr.models.base import AudioInput

TARGET_SR = 16000


class CanaryQwenSalmModel:
    """NVIDIA Canary-Qwen Speech-Augmented LM (SALM) adapter.

    Hybrid model: FastConformer audio encoder + Qwen3 LLM decoder, connected
    via LoRA adapters. Loaded through NeMo's speechlm2 namespace, transcription
    happens via the model's generate() with a chat-templated prompt and
    file-path audio references. Per-clip loop with a temp WAV per call — the
    upstream API doesn't accept in-memory arrays directly.
    """

    def __init__(
        self,
        hf_id: str,
        language: str | None = None,
        max_new_tokens: int = 256,
    ):
        import torch
        from nemo.collections.speechlm2.models import SALM

        self.id = f"canary-qwen:{hf_id}"
        self.model = SALM.from_pretrained(hf_id)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.language = language  # accepted but unused; model is English-only
        self.max_new_tokens = max_new_tokens
        self._tmp_dir = tempfile.mkdtemp(prefix="rasr-canary-qwen-")
        self._audio_locator = getattr(self.model, "audio_locator_tag", "<audio>")

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

        results: list[str] = []
        for i, a in enumerate(audios):
            arr = self._resample(a.array, a.sr)
            wav_path = Path(self._tmp_dir) / f"clip_{i:08d}.wav"
            sf.write(str(wav_path), arr, TARGET_SR, subtype="PCM_16")

            answer_ids = self.model.generate(
                prompts=[
                    [
                        {
                            "role": "user",
                            "content": (
                                f"Transcribe the following: {self._audio_locator}"
                            ),
                            "audio": [str(wav_path)],
                        }
                    ]
                ],
                max_new_tokens=self.max_new_tokens,
            )
            text = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
            results.append(text.strip())

            try:
                wav_path.unlink()
            except OSError:
                pass

        return results
