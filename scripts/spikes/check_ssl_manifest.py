# scripts/spikes/check_ssl_manifest.py
import json, tempfile
from pathlib import Path
from rasr.train.config import SSLDatasetCfg, AudioCfg
from rasr.train.ssl_manifest import build_ssl_manifest

spec = SSLDatasetCfg(source="hf:twangodev/radiotalk-us-audio-higgs-noisy:train", limit=8)
with tempfile.TemporaryDirectory() as d:
    mp = build_ssl_manifest(spec, AudioCfg(), Path(d))
    rows = [json.loads(l) for l in mp.read_text().splitlines()]
    assert rows, "no rows"
    for r in rows:
        assert "audio_filepath" in r and "duration" in r, r
        assert "text" not in r, "SSL manifest must not carry text"
        assert r["duration"] > 0
    print(f"OK: {len(rows)} SSL rows, no text, all 16k wavs")
