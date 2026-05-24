from __future__ import annotations

import numpy as np

from rasr.models.base import AudioInput

TARGET_SR = 16000

_TRANSCRIBE_PROMPT = (
    "<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>"
)


class Phi4MultimodalModel:
    """Microsoft Phi-4-Multimodal adapter (audio + text LLM).

    Phi-4-Multimodal is a 5.6B-param LLM with vision and speech LoRA adapters.
    Transcription uses a fixed chat-templated prompt with `<|audio_1|>` as the
    audio placeholder. Per-clip loop — the processor accepts in-memory numpy
    arrays so no temp WAVs needed, but batching across utterances requires
    padding handling we don't bother with for eval.

    Note: the model's custom modeling code (via trust_remote_code) was
    written against transformers ~4.49 + peft ~0.13. Newer versions break
    on multiple fronts: meta-tensor init in transformers 5.x, peft API
    drift on `prepare_inputs_for_generation`, and `num_logits_to_keep=None`
    handling. Pinning the full compat stack is left as an exercise; the
    adapter scaffold here is correct against the documented API but needs
    a frozen-env run to actually execute.
    """

    def __init__(
        self,
        hf_id: str,
        language: str | None = None,
        max_new_tokens: int = 256,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        self.id = f"phi4-multimodal:{hf_id}"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        # Transformers 5.x's lazy/meta-tensor init breaks Phi-4's custom
        # modeling code (which calls .item() during construction). Force
        # eager weight init.
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=self._dtype,
            trust_remote_code=True,
            _attn_implementation="eager",
            low_cpu_mem_usage=False,
        )
        if torch.cuda.is_available():
            self.model = self.model.to(self._device)
        self.model.eval()
        self.generation_config = GenerationConfig.from_pretrained(hf_id)
        self.language = language  # accepted but unused (English transcription only)
        self.max_new_tokens = max_new_tokens

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
        import torch

        results: list[str] = []
        for a in audios:
            arr = self._resample(a.array, a.sr)
            inputs = self.processor(
                text=_TRANSCRIBE_PROMPT,
                audios=[(arr, TARGET_SR)],
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    generation_config=self.generation_config,
                )
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
            text = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            results.append(text.strip())
        return results
