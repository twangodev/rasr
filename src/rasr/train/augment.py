"""Custom NeMo audio perturbations for ATC domain adaptation."""

from __future__ import annotations


class BandpassPerturbation:
    """Apply a Butterworth bandpass filter to the audio.

    Defaults to 300-3400 Hz, the VHF-channel envelope of real ATC radio.
    Used to close the spectral gap between full-bandwidth synthetic audio
    (e.g. radiotalk TTS) and real VHF transmissions.

    Inherits from NeMo's Perturbation by interface (callable on AudioSegment),
    but doesn't import NeMo at module-import time so it remains cheap to load.
    """

    def __init__(
        self,
        low_hz: int = 300,
        high_hz: int = 3400,
        sr: int = 16000,
        order: int = 4,
        prob: float = 1.0,
    ):
        from scipy.signal import butter

        self._prob = float(prob)
        self._low = low_hz
        self._high = high_hz
        self._sr = sr
        self._sos = butter(
            order, [low_hz, high_hz], btype="band", fs=sr, output="sos"
        )

    def perturb(self, data) -> None:
        import numpy as np
        from scipy.signal import sosfilt

        if np.random.random() >= self._prob:
            return
        samples = sosfilt(self._sos, data._samples).astype(np.float32)
        data._samples = samples


def register() -> None:
    """Register the bandpass perturbation with NeMo's factory so it's
    addressable via the augmentor dict (`augmentor.bandpass.{prob,low_hz,...}`).

    Safe to call multiple times.
    """
    from nemo.collections.asr.parts.preprocessing import perturb

    if not hasattr(perturb, "perturbation_types"):
        return
    perturb.perturbation_types.setdefault("bandpass", BandpassPerturbation)
