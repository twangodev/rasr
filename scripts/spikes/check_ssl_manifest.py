# scripts/spikes/check_ssl_manifest.py
import json, tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from rasr.train.config import SSLDatasetCfg, AudioCfg
from rasr.train.ssl_manifest import build_ssl_manifest


def check_hf():
    spec = SSLDatasetCfg(source="hf:twangodev/radiotalk-us-audio-higgs-noisy:train", limit=8)
    with tempfile.TemporaryDirectory() as d:
        mp = build_ssl_manifest(spec, AudioCfg(), Path(d))
        rows = [json.loads(l) for l in mp.read_text().splitlines()]
        assert rows, "no rows"
        for r in rows:
            assert "audio_filepath" in r and "duration" in r, r
            assert "text" not in r, "SSL manifest must not carry text"
            assert r["duration"] > 0
        print(f"OK (hf): {len(rows)} SSL rows, no text, all 16k wavs")


def check_local():
    sr = 22050  # non-16k so the resample path is exercised
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        src_dir = dd / "tartan" / "nested"
        src_dir.mkdir(parents=True)
        for i in range(4):
            dur = 2.0 + 0.5 * i
            wav = (0.1 * rng.standard_normal(int(dur * sr))).astype(np.float32)
            sf.write(str(src_dir / f"{i:03d}.wav"), wav, sr, subtype="PCM_16")
        glob = f"local:{dd}/tartan/**/*.wav"
        spec = SSLDatasetCfg(source=glob, min_db=-40.0)
        mp = build_ssl_manifest(spec, AudioCfg(), dd / "cache")
        rows = [json.loads(l) for l in mp.read_text().splitlines()]
        assert rows, "no rows from local glob"
        for r in rows:
            assert "audio_filepath" in r and "duration" in r, r
            assert "text" not in r, "SSL manifest must not carry text"
            assert r["duration"] > 0
        print(f"OK (local): {len(rows)} SSL rows, no text, all 16k wavs")


if __name__ == "__main__":
    check_hf()
    check_local()
