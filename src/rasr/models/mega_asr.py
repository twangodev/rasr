from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from rasr.models.base import AudioInput

TARGET_SR = 16000

# Mega-ASR is built on Qwen3-ASR-1.7B, which expects full language names
# (e.g. "English"), not ISO codes (e.g. "en"). Mirror the mapping in qwen.py.
_ISO_TO_NAME = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
}

# The upstream `xzf-thu/Mega-ASR` repo has no Python packaging metadata, so we
# vendor it as a git submodule under `third_party/Mega-ASR` (canonical upstream)
# and put its `src/` on sys.path at import time. `_PROJECT_ROOT` resolves
# relative to this file so it works both from the source checkout and from an
# installed editable wheel.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MEGAASR_SRC = _PROJECT_ROOT / "third_party" / "Mega-ASR" / "src"


def _ensure_megaasr_importable() -> None:
    if not _MEGAASR_SRC.is_dir():
        raise RuntimeError(
            "third_party/Mega-ASR submodule not found at "
            f"{_MEGAASR_SRC}. Run `git submodule update --init "
            "--recursive` to fetch it."
        )
    src_str = str(_MEGAASR_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


class MegaAsrModel:
    """Mega-ASR (Qwen3-ASR-1.7B + audio-quality router + LoRA delta) adapter.

    The upstream `MegaASR` class loads a Qwen3-ASR-1.7B backbone, an audio
    quality router that classifies each clip as clean/degraded, and a LoRA
    delta that's merged in for degraded inputs. The infer() API only accepts a
    file path, so we write each numpy array to a temp WAV at 16 kHz mono PCM16
    per clip (same shape as canary-qwen).

    The HF snapshot lays out checkpoints as:
        <snapshot>/Qwen3-ASR-1.7B/             (backbone weights + processor)
        <snapshot>/mega-asr-merged/            (LoRA adapter dir)
        <snapshot>/audio_quality_router/best_acc_model.safetensors
    """

    def __init__(
        self,
        hf_id: str,
        language: str | None = None,
        routing_enabled: bool = True,
        quality_threshold: float = 0.5,
        keep_delta_on_gpu: bool = False,
    ):
        _ensure_megaasr_importable()
        import torch
        from huggingface_hub import snapshot_download
        from MegaASR.model.megaASR import MegaASR as _MegaASR

        self.id = f"mega-asr:{hf_id}"
        self.language = _ISO_TO_NAME.get(language, language) if language else None

        ckpt_root = Path(snapshot_download(repo_id=hf_id))
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = _MegaASR(
            model_path=str(ckpt_root / "Qwen3-ASR-1.7B"),
            lora_dir=str(ckpt_root / "mega-asr-merged"),
            router_checkpoint=str(
                ckpt_root / "audio_quality_router" / "best_acc_model.safetensors"
            ),
            routing_enabled=routing_enabled,
            quality_threshold=quality_threshold,
            device_map=device_map,
            keep_delta_on_gpu=keep_delta_on_gpu,
        )

        self._tmp_dir = tempfile.mkdtemp(prefix="rasr-mega-asr-")

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == TARGET_SR:
            return audio.astype(np.float32)
        import librosa

        return librosa.resample(
            audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR
        )

    @staticmethod
    def _extract_text(result) -> str:
        # MegaASR.infer with return_route=True returns
        # {"text": <str|list[str]>, "use_lora": ..., "degraded_prob": ...,
        #  "route_source": ...}. Without return_route it's just the str/list.
        if isinstance(result, dict):
            result = result.get("text", "")
        if isinstance(result, list):
            result = result[0] if result else ""
        return str(result).strip()

    def transcribe(self, audios: list[AudioInput]) -> list[str]:
        if not audios:
            return []
        import soundfile as sf

        results: list[str] = []
        for i, a in enumerate(audios):
            arr = self._resample(a.array, a.sr)
            wav_path = Path(self._tmp_dir) / f"clip_{i:08d}.wav"
            sf.write(str(wav_path), arr, TARGET_SR, subtype="PCM_16")

            try:
                out = self.model.infer(str(wav_path), language=self.language)
                results.append(self._extract_text(out))
            finally:
                try:
                    wav_path.unlink()
                except OSError:
                    pass

        return results
