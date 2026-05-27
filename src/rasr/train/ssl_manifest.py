from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from rasr.train.config import AudioCfg, SSLDatasetCfg
from rasr.train.manifest import _hash_spec, _load_hf_streaming, _resample, _safe


def build_ssl_manifest(spec: SSLDatasetCfg, audio_cfg: AudioCfg, cache_dir: Path) -> Path:
    """Convert an unlabeled `hf:` source to a NeMo SSL manifest (no text).

    Audio is resampled to `audio_cfg.sample_rate` mono PCM16 and cached under
    `cache_dir/ssl__<repo>__<split>__<hash>/wavs/`. Clips outside
    [min_duration, max_duration], or below `spec.min_db` peak level, are
    dropped. Idempotent: returns early if the manifest already exists.
    """
    if not spec.source.startswith("hf:"):
        raise ValueError(
            f"build_ssl_manifest only supports hf: sources; got {spec.source!r}"
        )
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
    sr = audio_cfg.sample_rate
    written = 0
    with manifest_path.open("w") as mf:
        for i, row in enumerate(ds):
            if spec.limit is not None and i >= spec.limit:
                break
            audio = row["audio"]
            arr = np.asarray(audio["array"], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if spec.min_db is not None and arr.size:
                peak = float(np.max(np.abs(arr)))
                peak_db = 20.0 * np.log10(peak + 1e-9)
                if peak_db < spec.min_db:
                    continue
            arr = _resample(arr, int(audio["sampling_rate"]), sr)
            duration = float(len(arr)) / sr
            if duration < audio_cfg.min_duration or duration > audio_cfg.max_duration:
                continue
            wav_path = wav_dir / f"{i:08d}.wav"
            if not wav_path.exists():
                sf.write(str(wav_path), arr, sr, subtype="PCM_16")
            mf.write(
                json.dumps({"audio_filepath": str(wav_path.resolve()), "duration": duration})
                + "\n"
            )
            written += 1
    if written == 0:
        manifest_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"No clips written for {spec.source!r}; check filters"
        )
    return manifest_path
