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


def _segment_cache_key(spec: SSLDatasetCfg) -> str:
    """Cache key that also folds in segmentation params so toggling them rebuilds."""
    if not spec.segment:
        return _hash_spec(spec.source, spec.limit)
    sig = (
        f"{spec.source}|limit={spec.limit}|seg=1"
        f"|top_db={spec.segment_top_db}|min_s={spec.segment_min_s}"
        f"|max_s={spec.segment_max_s}|pad_s={spec.segment_pad_s}|min_db={spec.min_db}"
    )
    return _hash_spec(sig, None)


def _emit_segments(
    arr: np.ndarray,
    sr: int,
    spec: SSLDatasetCfg,
    wav_dir: Path,
    stem: str,
    mf,
) -> int:
    """Energy-VAD split one resampled-to-`sr` mono array into utterance chunks.

    Each non-silent interval (from `librosa.effects.split`) is padded, then chopped
    into windows no longer than `segment_max_s`. Chunks shorter than `segment_min_s`
    or below the `min_db` peak filter are dropped. Writes one PCM16 WAV per kept
    chunk as `<stem>_<seg_idx:04d>.wav` and appends a manifest row. Returns the
    number of chunks written.
    """
    import librosa

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    intervals = librosa.effects.split(arr, top_db=spec.segment_top_db)
    pad = int(spec.segment_pad_s * sr)
    win = int(spec.segment_max_s * sr)
    written = 0
    for s, e in intervals:
        s = max(0, s - pad)
        e = min(len(arr), e + pad)
        seg = arr[s:e]
        for k in range(0, len(seg), win):
            chunk = seg[k:k + win]
            dur = len(chunk) / sr
            if dur < spec.segment_min_s:
                continue
            if spec.min_db is not None and chunk.size:
                peak_db = 20.0 * np.log10(float(np.max(np.abs(chunk))) + 1e-9)
                if peak_db < spec.min_db:
                    continue
            wav_path = wav_dir / f"{stem}_{written:04d}.wav"
            if not wav_path.exists():
                sf.write(str(wav_path), chunk, sr, subtype="PCM_16")
            mf.write(
                json.dumps(
                    {"audio_filepath": str(wav_path.resolve()), "duration": dur}
                )
                + "\n"
            )
            written += 1
    return written


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

    cache_key = _segment_cache_key(spec)
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
            if spec.segment:
                arr = np.asarray(audio["array"], dtype=np.float32)
                arr = _resample(arr, int(audio["sampling_rate"]), audio_cfg.sample_rate)
                written += _emit_segments(
                    arr, audio_cfg.sample_rate, spec, wav_dir, f"{i:08d}", mf
                )
                continue
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
    cache_key = _segment_cache_key(spec)
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
            if spec.segment:
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                arr = _resample(arr, int(src_sr), audio_cfg.sample_rate)
                written += _emit_segments(
                    arr, audio_cfg.sample_rate, spec, wav_dir, f"{i:08d}", mf
                )
                continue
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
