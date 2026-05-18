from __future__ import annotations

from typing import Iterator

import numpy as np
from datasets import load_dataset as hf_load_dataset

from rasr.datasets.base import Utterance


class HfDataset:
    """Streaming adapter for a HuggingFace audio dataset.

    Expects an `audio` column (HF Audio feature) and a configurable text column
    for the reference transcript. Default is `text_normalized` to match the
    radiotalk-us-audio family of datasets.
    """

    def __init__(
        self,
        repo: str,
        split: str = "train",
        reference_field: str = "text_normalized",
        utt_id_fields: tuple[str, ...] = ("scenario_id", "turn_idx"),
        meta_fields: tuple[str, ...] = (
            "speaker",
            "profile",
            "effective_snr_db",
            "applied_effects",
        ),
    ):
        self.id = f"hf:{repo}:{split}"
        self.repo = repo
        self.split = split
        self.reference_field = reference_field
        self.utt_id_fields = utt_id_fields
        self.meta_fields = meta_fields
        self._ds = hf_load_dataset(repo, split=split, streaming=True)

    def __iter__(self) -> Iterator[Utterance]:
        for row in self._ds:
            audio = row["audio"]
            samples = np.asarray(audio["array"], dtype=np.float32)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            utt_id = "-".join(str(row[f]) for f in self.utt_id_fields if f in row)
            yield Utterance(
                utt_id=utt_id,
                audio=samples,
                sr=int(audio["sampling_rate"]),
                reference=row[self.reference_field],
                meta={f: row[f] for f in self.meta_fields if f in row},
            )
