from __future__ import annotations

import numpy as np

from rasr.models.base import AudioInput

TARGET_SR = 16000

_USER_PROMPT = (
    "<|audio|>transcribe the speech with proper punctuation and capitalization."
)


class GraniteSpeechModel:
    """IBM Granite Speech adapter.

    Granite Speech is an LLM with an audio adapter; transcription happens via a
    chat-templated prompt with an `<|audio|>` placeholder. Per-clip loop —
    batching across utterances is possible but requires padding handling that
    isn't worth the complexity for a v1 adapter.
    """

    def __init__(
        self,
        hf_id: str,
        max_new_tokens: int = 256,
    ):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.id = f"granite-speech:{hf_id}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            hf_id,
            device_map=device,
            torch_dtype=dtype,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": _USER_PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )

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
            wav = torch.from_numpy(audio_16k).unsqueeze(0)
            inputs = self.processor(
                self._chat_prompt,
                wav,
                device=self._device,
                return_tensors="pt",
            ).to(self._device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            generated = outputs[0, input_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text.strip())
        return results
