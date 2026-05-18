from __future__ import annotations

from rasr.models.base import AudioInput


class QwenAsrModel:
    def __init__(
        self,
        hf_id: str,
        language: str | None = None,
        max_new_tokens: int = 256,
        max_inference_batch_size: int = 32,
    ):
        import torch
        from qwen_asr import Qwen3ASRModel as _QwenModel

        self.id = f"qwen-asr:{hf_id}"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = _QwenModel.from_pretrained(
            hf_id,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        self.language = language

    def transcribe(self, audios: list[AudioInput]) -> list[str]:
        if not audios:
            return []
        results = self.model.transcribe(
            audio=[(a.array, a.sr) for a in audios],
            language=self.language,
        )
        return [r.text.strip() for r in results]
