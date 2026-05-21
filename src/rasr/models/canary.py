from __future__ import annotations

import numpy as np

from rasr.models.base import AudioInput

TARGET_SR = 16000


class CanaryAsrModel:
    """NVIDIA Canary multi-task ASR adapter (EncDecMultiTaskModel).

    Canary does both ASR and speech translation; the task is selected via
    source_lang/target_lang. For ASR we set them equal. pnc='yes' enables
    punctuation and capitalization in the output.

    Distinct from `canary-qwen:` (which is the SALM/LLM-decoder variant).
    """

    def __init__(
        self,
        hf_id: str,
        language: str | None = None,
        batch_size: int = 16,
        pnc: str = "yes",
    ):
        import nemo.collections.asr as nemo_asr
        import torch

        self.id = f"canary:{hf_id}"
        self.model = nemo_asr.models.ASRModel.from_pretrained(hf_id)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.language = language or "en"
        self.batch_size = batch_size
        self.pnc = pnc

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
            source_lang=self.language,
            target_lang=self.language,
            pnc=self.pnc,
            verbose=False,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return [
            (o.text if hasattr(o, "text") else str(o)).strip() for o in outputs
        ]
