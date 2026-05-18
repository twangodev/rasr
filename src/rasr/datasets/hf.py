from __future__ import annotations

from typing import Iterator

import numpy as np
from datasets import load_dataset as hf_load_dataset

from rasr.datasets.base import Utterance

_REFERENCE_CANDIDATES = (
    "text_normalized",
    "text",
    "transcript",
    "transcription",
    "sentence",
    "reference",
)

_UTT_ID_CANDIDATES = (
    ("scenario_id", "turn_idx", "variant_idx"),  # radiotalk
    ("scenario_id", "turn_idx"),
    ("id",),
    ("utt_id",),
    ("path",),
    ("audio_path",),
    ("file",),
)


class HfDataset:
    """Streaming adapter for a HuggingFace audio dataset.

    Auto-detects the reference text field and a stable per-row id. Defaults
    target the radiotalk-us-audio family but also work on jlvdoorn/atco2-asr,
    ATCOSIM, etc. without per-dataset configuration.
    """

    def __init__(
        self,
        repo: str,
        split: str = "train",
        reference_field: str | None = None,
        utt_id_fields: tuple[str, ...] | None = None,
        meta_fields: tuple[str, ...] = (
            "speaker",
            "profile",
            "effective_snr_db",
            "applied_effects",
            "info",
        ),
    ):
        self.id = f"hf:{repo}:{split}"
        self.repo = repo
        self.split = split
        self.reference_field = reference_field  # may be resolved on first row
        self.utt_id_fields = utt_id_fields  # may be resolved on first row
        self.meta_fields = meta_fields
        self._ds = hf_load_dataset(repo, split=split, streaming=True)

    def _resolve_fields(self, row: dict) -> None:
        if self.reference_field is None:
            for cand in _REFERENCE_CANDIDATES:
                if cand in row:
                    self.reference_field = cand
                    break
            if self.reference_field is None:
                raise KeyError(
                    f"No reference text column found in {self.id}; "
                    f"tried {_REFERENCE_CANDIDATES}"
                )
        if self.utt_id_fields is None:
            for candidate_set in _UTT_ID_CANDIDATES:
                if any(f in row for f in candidate_set):
                    self.utt_id_fields = candidate_set
                    break
            if self.utt_id_fields is None:
                self.utt_id_fields = ()  # fall back to enumerate index

    def __iter__(self) -> Iterator[Utterance]:
        for i, row in enumerate(self._ds):
            if self.reference_field is None or self.utt_id_fields is None:
                self._resolve_fields(row)
            audio = row["audio"]
            samples = np.asarray(audio["array"], dtype=np.float32)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            parts = [str(row[f]) for f in self.utt_id_fields if f in row]
            utt_id = "-".join(parts) if parts else f"row-{i:06d}"
            yield Utterance(
                utt_id=utt_id,
                audio=samples,
                sr=int(audio["sampling_rate"]),
                reference=row[self.reference_field],
                meta={f: row[f] for f in self.meta_fields if f in row},
            )
