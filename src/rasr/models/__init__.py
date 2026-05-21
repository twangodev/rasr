from __future__ import annotations

from rasr.models.base import Model


def load_model(model_id: str, *, language: str | None = None) -> Model:
    """Resolve a model spec to a Model instance.

    `language` is an ISO 639-1 code ('en', 'de', ...) or None for auto-detect.
    Each adapter maps the code to its model's preferred format internally.
    """
    scheme, _, ref = model_id.partition(":")
    if not ref:
        raise ValueError(f"Model ID must be '<scheme>:<ref>', got: {model_id!r}")

    if scheme == "whisper":
        from rasr.models.whisper import WhisperModel

        return WhisperModel(ref, language=language)

    if scheme == "qwen-asr":
        from rasr.models.qwen import QwenAsrModel

        return QwenAsrModel(ref, language=language)

    if scheme == "cohere":
        from rasr.models.cohere import CohereAsrModel

        return CohereAsrModel(ref, language=language)

    if scheme == "granite-speech":
        from rasr.models.granite_speech import GraniteSpeechModel

        return GraniteSpeechModel(ref, language=language)

    if scheme == "parakeet":
        from rasr.models.parakeet import ParakeetModel

        return ParakeetModel(ref, language=language)

    if scheme == "nemo":
        from rasr.models.parakeet import ParakeetModel

        return ParakeetModel(ref, language=language, from_path=True)

    if scheme == "canary-qwen":
        from rasr.models.canary_qwen import CanaryQwenSalmModel

        return CanaryQwenSalmModel(ref, language=language)

    if scheme == "canary":
        from rasr.models.canary import CanaryAsrModel

        return CanaryAsrModel(ref, language=language)

    if scheme == "mega-asr":
        from rasr.models.mega_asr import MegaAsrModel

        return MegaAsrModel(ref, language=language)

    raise ValueError(f"Unknown model scheme: {scheme!r}")


__all__ = ["Model", "load_model"]
