from __future__ import annotations

from pathlib import Path

from rasr.datasets.base import Dataset, Utterance
from rasr.datasets.manifest import ManifestDataset


def load_dataset(path: Path) -> Dataset:
    return ManifestDataset(Path(path))


__all__ = ["Dataset", "Utterance", "ManifestDataset", "load_dataset"]
