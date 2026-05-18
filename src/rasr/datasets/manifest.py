from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import soundfile as sf

from rasr.datasets.base import Utterance


class ManifestDataset:
    """Dataset backed by a JSONL manifest.

    Each line is a JSON object with fields:
        utt_id:    string identifier for the utterance
        audio:     path to an audio file (relative paths resolve against
                   the manifest's parent directory)
        reference: ground-truth transcript
        meta:      optional dict of arbitrary metadata
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.id = f"manifest:{self.path.name}"
        text = self.path.read_text()
        self._entries = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
        self._root = self.path.parent

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Utterance]:
        for e in self._entries:
            audio_field = Path(e["audio"])
            audio_path = audio_field if audio_field.is_absolute() else self._root / audio_field
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            yield Utterance(
                utt_id=e["utt_id"],
                audio=audio,
                sr=sr,
                reference=e["reference"],
                meta=e.get("meta", {}),
            )
