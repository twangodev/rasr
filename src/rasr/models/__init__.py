from __future__ import annotations

from rasr.models.base import Model


def load_model(model_id: str) -> Model:
    scheme, _, ref = model_id.partition(":")
    if not ref:
        raise ValueError(f"Model ID must be '<scheme>:<ref>', got: {model_id!r}")

    if scheme == "whisper":
        from rasr.models.whisper import WhisperModel

        return WhisperModel(ref)

    raise ValueError(f"Unknown model scheme: {scheme!r}")


__all__ = ["Model", "load_model"]
