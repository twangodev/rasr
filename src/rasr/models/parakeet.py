from __future__ import annotations

import numpy as np

from rasr.models.base import AudioInput

TARGET_SR = 16000


class ParakeetModel:
    """NVIDIA Parakeet (NeMo) ASR adapter.

    Parakeet ships through NeMo, not transformers. The model accepts a list of
    in-memory numpy arrays at 16kHz mono and does its own internal batching.
    Language auto-detected — the upstream API doesn't take a language hint, so
    the constructor ignores it for now.

    `from_path=True` loads a locally saved .nemo checkpoint via
    `ASRModel.restore_from`, used by the `nemo:` model scheme to consume
    rasr-trained outputs.
    """

    def __init__(
        self,
        hf_id_or_path: str,
        language: str | None = None,
        batch_size: int = 16,
        from_path: bool = False,
    ):
        import nemo.collections.asr as nemo_asr
        import torch

        if from_path:
            self.id = f"nemo:{hf_id_or_path}"
            self.model = nemo_asr.models.ASRModel.restore_from(hf_id_or_path)
        else:
            self.id = f"parakeet:{hf_id_or_path}"
            self.model = nemo_asr.models.ASRModel.from_pretrained(hf_id_or_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.language = language  # accepted but unused; Parakeet auto-detects
        self.batch_size = batch_size

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
        arrays = [self._resample(a.array, a.sr) for a in audios]
        outputs = self.model.transcribe(
            audio=arrays,
            batch_size=min(self.batch_size, len(arrays)),
            verbose=False,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return [
            (o.text if hasattr(o, "text") else str(o)).strip() for o in outputs
        ]
