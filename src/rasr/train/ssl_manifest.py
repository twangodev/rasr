from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from rasr.train.config import AudioCfg, SSLDatasetCfg
from rasr.train.manifest import _hash_spec, _load_hf_streaming, _resample, _safe


def _process_clip(
    arr: np.ndarray,
    src_sr: int,
    audio_cfg: AudioCfg,
    min_db: float | None,
    wav_path: Path,
) -> float | None:
    """Filter, resample, and cache one clip; return its duration or None if dropped.

    Applies the shared SSL ingest pipeline: downmix to mono, min_db peak-level
    filter, resample to `audio_cfg.sample_rate`, [min_duration, max_duration]
    filter, and write a PCM16 WAV to `wav_path` (skipped if already present).
    Returns the clip duration in seconds, or None if it was filtered out.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if min_db is not None and arr.size:
        peak = float(np.max(np.abs(arr)))
        peak_db = 20.0 * np.log10(peak + 1e-9)
        if peak_db < min_db:
            return None
    sr = audio_cfg.sample_rate
    arr = _resample(arr, src_sr, sr)
    duration = float(len(arr)) / sr
    if duration < audio_cfg.min_duration or duration > audio_cfg.max_duration:
        return None
    if not wav_path.exists():
        sf.write(str(wav_path), arr, sr, subtype="PCM_16")
    return duration


def build_ssl_manifest(spec: SSLDatasetCfg, audio_cfg: AudioCfg, cache_dir: Path) -> Path:
    """Convert an unlabeled audio source to a NeMo SSL manifest (no text).

    `spec.source` is either `hf:<owner>/<repo>[:<split>]` (streamed from the Hub)
    or `local:<glob>` (e.g. `local:/data/tartan/**/*.wav`, globbed recursively).
    Audio is resampled to `audio_cfg.sample_rate` mono PCM16 and cached under
    `cache_dir/ssl__.../wavs/`. Clips outside [min_duration, max_duration], or
    below `spec.min_db` peak level, are dropped. Idempotent: returns early if the
    manifest already exists.
    """
    if spec.source.startswith("hf:"):
        return _build_from_hf(spec, audio_cfg, cache_dir)
    if spec.source.startswith("local:"):
        return _build_from_local(spec, audio_cfg, cache_dir)
    raise ValueError(
        f"build_ssl_manifest supports hf: and local: sources; got {spec.source!r}"
    )


def _build_from_hf(spec: SSLDatasetCfg, audio_cfg: AudioCfg, cache_dir: Path) -> Path:
    rest = spec.source[len("hf:"):]
    repo, _, split = rest.partition(":")
    split = split or "train"

    cache_key = _hash_spec(spec.source, spec.limit)
    out_dir = cache_dir / f"ssl__{_safe(repo)}__{split}__{cache_key}"
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        return manifest_path

    ds = _load_hf_streaming(repo, split)
    written = 0
    with manifest_path.open("w") as mf:
        for i, row in enumerate(ds):
            if spec.limit is not None and i >= spec.limit:
                break
            audio = row["audio"]
            wav_path = wav_dir / f"{i:08d}.wav"
            duration = _process_clip(
                np.asarray(audio["array"], dtype=np.float32),
                int(audio["sampling_rate"]),
                audio_cfg,
                spec.min_db,
                wav_path,
            )
            if duration is None:
                continue
            mf.write(
                json.dumps({"audio_filepath": str(wav_path.resolve()), "duration": duration})
                + "\n"
            )
            written += 1
    if written == 0:
        manifest_path.unlink(missing_ok=True)
        raise RuntimeError(f"No clips written for {spec.source!r}; check filters")
    return manifest_path


def _build_from_local(spec: SSLDatasetCfg, audio_cfg: AudioCfg, cache_dir: Path) -> Path:
    pattern = spec.source[len("local:"):]
    cache_key = _hash_spec(spec.source, spec.limit)
    out_dir = cache_dir / f"ssl__local__{cache_key}"
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        return manifest_path

    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise RuntimeError(f"No files matched glob {pattern!r} (from {spec.source!r})")

    written = 0
    with manifest_path.open("w") as mf:
        for i, src in enumerate(files):
            if spec.limit is not None and i >= spec.limit:
                break
            arr, src_sr = sf.read(src, dtype="float32", always_2d=False)
            wav_path = wav_dir / f"{i:08d}.wav"
            duration = _process_clip(arr, int(src_sr), audio_cfg, spec.min_db, wav_path)
            if duration is None:
                continue
            mf.write(
                json.dumps({"audio_filepath": str(wav_path.resolve()), "duration": duration})
                + "\n"
            )
            written += 1
    if written == 0:
        manifest_path.unlink(missing_ok=True)
        raise RuntimeError(f"No clips written for {spec.source!r}; check filters")
    return manifest_path
