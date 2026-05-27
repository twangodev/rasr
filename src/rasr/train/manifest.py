from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset as hf_load_dataset

from rasr.train.config import AudioCfg, DatasetCfg

_TEXT_CANDIDATES = (
    "text_normalized",
    "text",
    "transcript",
    "transcription",
    "sentence",
    "reference",
)


def _hash_spec(spec: str, limit: int | None) -> str:
    sig = f"{spec}|limit={limit}"
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _resample(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return arr
    import librosa

    return librosa.resample(arr.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr)


def _load_hf_streaming(repo: str, split: str):
    """Stream an HF dataset split, with an offline fallback to the hub cache.

    The normal path (`load_dataset(repo, streaming=True)`) works online and for
    any repo previously materialized by `datasets` (it has a cached builder
    module). A repo pulled only via `hf download` lives in the hub *snapshot*
    cache with no builder module, so offline resolution raises ConnectionError.
    In that case we resolve the cached snapshot via `snapshot_download(...,
    local_files_only=True)` and stream its parquet shards directly. This keeps
    fresh-clone (online) behavior unchanged while letting an as-yet-unpushed
    dataset be trained from cache; it assumes the snapshot's parquet files are
    all the requested split (true for single-split radiotalk corpora).
    """
    try:
        return hf_load_dataset(repo, split=split, streaming=True)
    except ConnectionError:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(
            repo,
            repo_type="dataset",
            allow_patterns=["*.parquet"],
            local_files_only=True,
        )
        files = sorted(glob.glob(os.path.join(snap, "**", "*.parquet"), recursive=True))
        if not files:
            raise RuntimeError(
                f"offline fallback for {repo!r}: no parquet shards under {snap}"
            )
        return hf_load_dataset(
            "parquet", data_files=files, split="train", streaming=True
        )


def build_manifest(
    spec: DatasetCfg,
    audio_cfg: AudioCfg,
    cache_dir: Path,
) -> Path:
    """Convert an `hf:owner/repo[:split]` spec to a NeMo JSONL manifest.

    Audio is resampled to `audio_cfg.sample_rate` mono PCM16 and cached under
    `cache_dir/<hashed-spec>/wavs/`. Idempotent: re-runs skip rows whose
    cached WAV already exists, and skip the manifest write entirely if it's
    already present and complete.

    Clips outside [min_duration, max_duration] are dropped.
    """
    if not spec.dataset.startswith("hf:"):
        raise ValueError(
            f"build_manifest only supports hf:<owner>/<repo>[:<split>] specs; "
            f"got: {spec.dataset!r}"
        )

    rest = spec.dataset[len("hf:") :]
    repo, _, split = rest.partition(":")
    split = split or "train"

    cache_key = _hash_spec(spec.dataset, spec.limit)
    out_dir = cache_dir / f"{_safe(repo)}__{split}__{cache_key}"
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    if manifest_path.exists():
        return manifest_path

    ds = _load_hf_streaming(repo, split)
    target_sr = audio_cfg.sample_rate
    min_dur = audio_cfg.min_duration
    max_dur = audio_cfg.max_duration

    written = 0
    text_field: str | None = None
    with manifest_path.open("w") as mf:
        for i, row in enumerate(ds):
            if spec.limit is not None and i >= spec.limit:
                break
            if text_field is None:
                for cand in _TEXT_CANDIDATES:
                    if cand in row:
                        text_field = cand
                        break
                if text_field is None:
                    raise KeyError(
                        f"No reference text column in {spec.dataset!r}; "
                        f"tried {_TEXT_CANDIDATES}"
                    )

            text = (row[text_field] or "").strip()
            if not text:
                continue

            audio = row["audio"]
            arr = np.asarray(audio["array"], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            arr = _resample(arr, int(audio["sampling_rate"]), target_sr)
            duration = float(len(arr)) / target_sr
            if duration < min_dur or duration > max_dur:
                continue

            wav_path = wav_dir / f"{i:08d}.wav"
            if not wav_path.exists():
                sf.write(str(wav_path), arr, target_sr, subtype="PCM_16")

            mf.write(
                json.dumps(
                    {
                        "audio_filepath": str(wav_path.resolve()),
                        "duration": duration,
                        "text": text,
                    }
                )
                + "\n"
            )
            written += 1

    if written == 0:
        manifest_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"No clips written for {spec.dataset!r}; check filters / text field"
        )

    return manifest_path
