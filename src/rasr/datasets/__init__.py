from __future__ import annotations

from pathlib import Path

from rasr.datasets.base import Dataset, Utterance
from rasr.datasets.manifest import ManifestDataset


def load_dataset(spec: str) -> Dataset:
    """Resolve a dataset spec to a Dataset instance.

    Supported forms:
        hf:<owner>/<repo>[:<split>]   — HuggingFace streaming dataset
        <path-to-manifest.jsonl>      — local JSONL manifest
    """
    s = str(spec)
    if s.startswith("hf:"):
        from rasr.datasets.hf import HfDataset

        rest = s[len("hf:") :]
        repo, _, split = rest.partition(":")
        return HfDataset(repo, split=split or "train")

    return ManifestDataset(Path(s))


__all__ = ["Dataset", "Utterance", "ManifestDataset", "load_dataset"]
