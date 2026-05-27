# SSL Continued-Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt `nvidia/parakeet-tdt-0.6b-v3`'s encoder to real US-ATC VHF acoustics via NEST-style continued SSL pretraining on unlabeled TartanAviation audio, then load the adapted encoder into the existing supervised Higgs finetune.

**Architecture:** Three stages bolted onto the existing `rasr train` pipeline — (1) build a no-text SSL manifest from TartanAviation, (2) continue-pretrain the Parakeet encoder with NeMo's SSL framework, (3) load the adapted encoder into the existing finetune. A Stage-0 spike pins down the NeMo SSL API and the encoder weight round-trip before any long run.

**Tech Stack:** NeMo (`nemo.collections.asr`, `examples/asr/speech_pretraining`), PyTorch Lightning, OmegaConf/Pydantic configs, Typer CLI, HF `datasets`/`huggingface_hub`.

---

## Reference reading (do once before starting)

- Spec: `docs/superpowers/specs/2026-05-26-ssl-continued-pretraining-design.md`
- Existing manifest builder to mirror: `src/rasr/train/manifest.py` (note the
  `_load_hf_streaming` offline fallback and the `build_manifest` filter loop).
- Existing trainer to mirror: `src/rasr/train/nemo.py`.
- Config schema: `src/rasr/train/config.py` (Pydantic `TrainConfig`, `DatasetCfg`).
- CLI wiring: `src/rasr/cli.py`, `src/rasr/train/cli.py`.
- NeMo SSL anchors to confirm in Stage 0: model classes under
  `nemo.collections.asr.models` (NEST denoising SSL model; e.g.
  `EncDecDenoiseMaskedTokenPredModel`) and the example configs/scripts in
  `examples/asr/speech_pretraining/`. `maybe_init_from_pretrained_checkpoint`
  / `init_from_nemo_model` is the encoder-transfer mechanism.

**Testing note:** this repo has no pytest harness and training touches GPU +
remote model downloads, so "tests" here are runnable verification scripts and
short smoke runs (the project's existing convention). Pure-logic pieces (the
SSL manifest builder) get a small standalone assertion script. Do NOT scaffold
a pytest suite unless a later task explicitly says so.

---

## Stage 0 — Verification spike (DO FIRST; gates everything)

### Task 0: Pin the NeMo SSL API and prove the encoder round-trip

**Files:**
- Create: `scripts/spikes/ssl_roundtrip_spike.py`

- [ ] **Step 1: Find the SSL model class and example config**

Run:
```bash
python -c "import nemo.collections.asr as a; import inspect, os; print(os.path.dirname(a.__file__))"
ls $(python -c "import nemo, os; print(os.path.dirname(os.path.dirname(nemo.__file__)))")/examples/asr/speech_pretraining/ 2>/dev/null || \
  python - <<'PY'
import nemo.collections.asr.models as m
print([n for n in dir(m) if 'SelfSup' in n or 'Denoise' in n or 'SSL' in n or 'MaskedTokenPred' in n])
PY
```
Expected: a model class name (e.g. `EncDecDenoiseMaskedTokenPredModel` and/or
`SpeechEncDecSelfSupervisedModel`) and the example SSL config dir. Record the
exact class name + config path in a comment at the top of the spike script.

- [ ] **Step 2: Write the spike script**

The spike: load Parakeet, capture its encoder config + state_dict, build an SSL
model whose encoder matches, load Parakeet encoder weights in, run a few SSL
steps on random/sample audio, save, then load the saved encoder back into a
Parakeet ASR model and assert no shape errors.

```python
# scripts/spikes/ssl_roundtrip_spike.py
# Spike: prove Parakeet-encoder -> NEST SSL -> reload-into-Parakeet round-trips.
# Fill the SSL model class + cfg path discovered in Step 1.
from __future__ import annotations
import tempfile, torch
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

REF = "nvidia/parakeet-tdt-0.6b-v3"

def main():
    base = nemo_asr.models.ASRModel.from_pretrained(REF, map_location="cpu")
    enc_cfg = base.cfg.encoder
    print("encoder _target_:", enc_cfg.get("_target_"))
    print("d_model:", enc_cfg.get("d_model"), "n_layers:", enc_cfg.get("n_layers"))
    enc_sd = base.encoder.state_dict()
    print("encoder params:", sum(v.numel() for v in enc_sd.values()))

    # --- build SSL model with a matching encoder (class from Step 1) ---
    # SSLModel = nemo_asr.models.<DISCOVERED_CLASS>
    # ssl_cfg = OmegaConf.load("<discovered example ssl cfg>")
    # ssl_cfg.model.encoder = enc_cfg           # force-match the encoder
    # ssl = SSLModel(cfg=ssl_cfg.model)
    # missing, unexpected = ssl.encoder.load_state_dict(enc_sd, strict=False)
    # print("SSL encoder load -> missing:", len(missing), "unexpected:", len(unexpected))
    # assert len(unexpected) == 0, unexpected

    # --- save just the encoder, reload into a fresh Parakeet ---
    # with tempfile.TemporaryDirectory() as d:
    #     ssl.save_to(f"{d}/ssl.nemo")
    #     tgt = nemo_asr.models.ASRModel.from_pretrained(REF, map_location="cpu")
    #     m2, u2 = tgt.encoder.load_state_dict(ssl.encoder.state_dict(), strict=False)
    #     assert len(u2) == 0, u2
    #     print("ROUND-TRIP OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the spike, iterate until ROUND-TRIP OK**

Run: `HF_HUB_OFFLINE=0 python scripts/spikes/ssl_roundtrip_spike.py`
Expected: prints encoder target/d_model, `SSL encoder load -> missing: 0 unexpected: 0`, `ROUND-TRIP OK`.

Decision gate: if the encoder configs cannot be aligned (unexpected keys / shape
mismatch that can't be resolved by copying `base.cfg.encoder` into the SSL cfg),
STOP and record the failure mode; fall back per spec (freeze-encoder variant, or
NEST-large-sized encoder). Do not proceed to Stage 2 until this passes.

- [ ] **Step 4: Commit the spike + findings**

```bash
git add scripts/spikes/ssl_roundtrip_spike.py
git commit -m "spike(ssl): prove Parakeet-encoder <-> NEST SSL weight round-trip"
```

---

## Stage 1 — Unlabeled SSL manifest

### Task 1: Add an unlabeled-source config type

**Files:**
- Modify: `src/rasr/train/config.py` (add `SSLDatasetCfg` + `SSLConfig`)

- [ ] **Step 1: Add the SSL config models**

Append to `src/rasr/train/config.py` (after `DatasetCfg`):

```python
class SSLDatasetCfg(BaseModel):
    """An unlabeled audio source for SSL pretraining.

    `source` is either `hf:<owner>/<repo>[:<split>]` or a local glob of audio
    files. No text/transcript is required or read.
    """
    source: str
    limit: int | None = None
    min_db: float | None = None  # keep only clips with max level >= this (TartanAviation: -20)
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from rasr.train.config import SSLDatasetCfg; print(SSLDatasetCfg(source='hf:x/y'))"`
Expected: prints `source='hf:x/y' limit=None min_db=None`.

- [ ] **Step 3: Commit**

```bash
git add src/rasr/train/config.py
git commit -m "feat(train): SSLDatasetCfg for unlabeled SSL sources"
```

### Task 2: Build a no-text SSL manifest

**Files:**
- Create: `src/rasr/train/ssl_manifest.py`
- Create: `scripts/spikes/check_ssl_manifest.py` (verification, not pytest)

- [ ] **Step 1: Write the verification script first (defines expected behavior)**

```python
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
```

- [ ] **Step 2: Run it to confirm it fails (module missing)**

Run: `HF_HUB_OFFLINE=1 python scripts/spikes/check_ssl_manifest.py`
Expected: `ModuleNotFoundError: rasr.train.ssl_manifest`.

- [ ] **Step 3: Implement `build_ssl_manifest`**

Mirror `build_manifest` but drop all text handling and add a `min_db` filter.
Reuse `_load_hf_streaming`, `_hash_spec`, `_safe`, `_resample` from
`manifest.py` (import them).

```python
# src/rasr/train/ssl_manifest.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from rasr.train.config import AudioCfg, SSLDatasetCfg
from rasr.train.manifest import _hash_spec, _safe, _resample, _load_hf_streaming


def build_ssl_manifest(spec: SSLDatasetCfg, audio_cfg: AudioCfg, cache_dir: Path) -> Path:
    """Convert an unlabeled `hf:` source to a NeMo SSL manifest (no text).

    Audio is resampled to `audio_cfg.sample_rate` mono PCM16 and cached. Clips
    outside [min_duration, max_duration], or below `spec.min_db` peak level, are
    dropped. Idempotent: returns early if the manifest already exists.
    """
    if not spec.source.startswith("hf:"):
        raise ValueError(f"build_ssl_manifest only supports hf: sources; got {spec.source!r}")
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
            mf.write(json.dumps({"audio_filepath": str(wav_path.resolve()), "duration": duration}) + "\n")
            written += 1
    if written == 0:
        manifest_path.unlink(missing_ok=True)
        raise RuntimeError(f"No clips written for {spec.source!r}; check filters")
    return manifest_path
```

- [ ] **Step 4: Run the verification script, expect PASS**

Run: `HF_HUB_OFFLINE=1 python scripts/spikes/check_ssl_manifest.py`
Expected: `OK: N SSL rows, no text, all 16k wavs`.

- [ ] **Step 5: Commit**

```bash
git add src/rasr/train/ssl_manifest.py scripts/spikes/check_ssl_manifest.py
git commit -m "feat(train): no-text SSL manifest builder (build_ssl_manifest)"
```

### Task 3: Wire TartanAviation as a reproducible source

**Files:**
- Modify: `src/rasr/train/ssl_manifest.py` (extend `_load_hf_streaming` use if Tartan is non-HF)
- Create: `docs/data-tartanaviation.md`

- [ ] **Step 1: Determine TartanAviation access path**

Run (investigate): check whether TartanAviation speech is mirrored on HF Hub
(`twangodev/...` or CMU AirLab) vs only at theairlab.org.
```bash
python - <<'PY'
from huggingface_hub import HfApi
print([d.id for d in HfApi().list_datasets(search="tartanaviation")][:10])
PY
```
Expected: either an HF dataset id (use it as `hf:<id>`), or empty (then document
the theairlab.org download + a local-glob ingest path in a follow-up).

- [ ] **Step 2: Write the data doc**

Create `docs/data-tartanaviation.md` recording: source URL, CC BY 4.0, 44.1 kHz,
~478 hr usable (>= -20 dB), how to obtain it, and the exact `source:`/`min_db:`
to use. (Keeps it reproducible per the project rule.)

- [ ] **Step 3: Commit**

```bash
git add docs/data-tartanaviation.md src/rasr/train/ssl_manifest.py
git commit -m "docs: TartanAviation SSL data source + ingest path"
```

---

## Stage 2 — Continued SSL pretraining

### Task 4: SSL recipe config

**Files:**
- Create: `configs/train/rtx6kpro/parakeet-ssl-tartan.yaml`

- [ ] **Step 1: Write the recipe**

Use the exact SSL model `_target_` + required cfg keys discovered in Task 0.
Skeleton (fill model section from Stage 0 findings):

```yaml
# Continued NEST SSL pretraining of parakeet-tdt-0.6b-v3's encoder on
# unlabeled TartanAviation (real US ATC VHF). Encoder is initialized from
# Parakeet; see docs/superpowers/specs/2026-05-26-ssl-continued-pretraining-design.md
defaults: [base, rtx6kpro/hw]
name: parakeet-ssl-tartan
ssl:
  source: hf:<tartanaviation-id>:train   # from Task 3; or local-glob path
  min_db: -20.0
  init_encoder_from: nvidia/parakeet-tdt-0.6b-v3
trainer:
  max_steps: 20000          # continued, not from-scratch; tune after smoke
  val_check_interval: 2000
output:
  dir: ckpt/${name}
```

- [ ] **Step 2: Commit**

```bash
git add configs/train/rtx6kpro/parakeet-ssl-tartan.yaml
git commit -m "feat(train): NEST SSL recipe on TartanAviation"
```

### Task 5: SSL training runner

**Files:**
- Create: `src/rasr/train/ssl.py`
- Modify: `src/rasr/train/config.py` (add `SSLTrainConfig`)
- Modify: `src/rasr/train/cli.py` (add `ssl` command)

- [ ] **Step 1: Add `SSLTrainConfig`**

In `config.py`, add a config that reuses `TrainerCfg`/`AudioCfg` and embeds the
SSL block:

```python
class SSLBlockCfg(BaseModel):
    source: str
    min_db: float | None = None
    init_encoder_from: str | None = None  # parakeet ref whose encoder seeds SSL

class SSLTrainConfig(BaseModel):
    name: str
    ssl: SSLBlockCfg
    audio: AudioCfg = Field(default_factory=AudioCfg)
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    scheduler: SchedulerCfg = Field(default_factory=SchedulerCfg)
    trainer: TrainerCfg
    output: OutputCfg
```
Add a `load_ssl_train_config` mirroring `load_train_config` (same
`_merge_with_defaults` + override + validate, returning `SSLTrainConfig`).

- [ ] **Step 2: Implement the runner**

`src/rasr/train/ssl.py`, mirroring `nemo.py:run`: build the SSL manifest via
`build_ssl_manifest`, construct the SSL model (class from Task 0), init its
encoder from `init_encoder_from` (the verified round-trip), set up the SSL
train dataloader (`manifest_filepath`, no labels), optimizer/scheduler from cfg,
a `pl.Trainer`, `trainer.fit(model)`, then `model.save_to(output/final.nemo)`.
Use the Stage-0 spike code as the proven basis for the encoder-init lines.

- [ ] **Step 3: Add the CLI command**

In `src/rasr/train/cli.py`, add an `ssl` command paralleling `run`:
```python
@app.command()
def ssl(config: Annotated[Path, typer.Option("--config","-c", exists=True, dir_okay=False)],
        dry_run: bool = typer.Option(False, "--dry-run"),
        overrides: Annotated[list[str] | None, typer.Argument()] = None) -> None:
    """Continue-pretrain an encoder with SSL on unlabeled audio."""
    from rasr.train.config import load_ssl_train_config
    cfg = load_ssl_train_config(config, overrides=overrides)
    console.print(f"[green]Loaded SSL config:[/green] {cfg.name}")
    console.print(f"  source: {cfg.ssl.source}  init_from: {cfg.ssl.init_encoder_from}")
    console.print(f"  steps:  {cfg.trainer.max_steps}")
    if dry_run:
        console.print("[yellow]--dry-run: skipping trainer.[/yellow]"); return
    from rasr.train.ssl import run as ssl_run
    console.print(f"[green]SSL done.[/green] Saved: {ssl_run(cfg)}")
```

- [ ] **Step 4: Verify dry-run wiring**

Run: `HF_HUB_OFFLINE=1 rasr train ssl -c configs/train/rtx6kpro/parakeet-ssl-tartan.yaml --dry-run`
Expected: prints the SSL config summary, no trainer.

- [ ] **Step 5: Commit**

```bash
git add src/rasr/train/ssl.py src/rasr/train/config.py src/rasr/train/cli.py
git commit -m "feat(train): rasr train ssl — NEST continued-pretraining runner"
```

### Task 6: SSL smoke run

- [ ] **Step 1: Smoke the full SSL loop on a tiny subset**

Run:
```bash
HF_HUB_OFFLINE=1 rasr train ssl -c configs/train/rtx6kpro/parakeet-ssl-tartan.yaml \
  ssl.source='hf:twangodev/radiotalk-us-audio-higgs-noisy:train' \
  trainer.max_steps=50 trainer.val_check_interval=25 \
  output.dir=ckpt/ssl-smoke 2>&1 | tail -30
```
Expected: encoder initializes from Parakeet (no shape errors — proven in Task 0),
SSL loss prints and trends down over 50 steps, `final.nemo` saved.

- [ ] **Step 2: Confirm the SSL loss decreased and a checkpoint exists**

Run: `ls -la ckpt/ssl-smoke/*.nemo`
Expected: a saved `.nemo`. (No commit — smoke artifact; `ckpt/` is gitignored.)

---

## Stage 3 — Supervised finetune from the adapted encoder

### Task 7: Let the finetune init its encoder from an SSL checkpoint

**Files:**
- Modify: `src/rasr/train/config.py` (add optional `model.init_encoder_from`)
- Modify: `src/rasr/train/nemo.py` (load encoder weights after `from_pretrained`)

- [ ] **Step 1: Add the field**

In `ModelCfg`:
```python
class ModelCfg(BaseModel):
    scheme: str
    ref: str
    language: str | None = None
    init_encoder_from: str | None = None  # path to an SSL .nemo whose encoder weights to load
```

- [ ] **Step 2: Load the encoder in the runner**

In `nemo.py:run`, immediately after `model = nemo_asr.models.ASRModel.from_pretrained(cfg.model.ref)`:
```python
if cfg.model.init_encoder_from:
    src = nemo_asr.models.ASRModel.restore_from(cfg.model.init_encoder_from, map_location="cpu")
    missing, unexpected = model.encoder.load_state_dict(src.encoder.state_dict(), strict=False)
    print(f"[rasr.train] loaded SSL encoder from {cfg.model.init_encoder_from} "
          f"(missing={len(missing)} unexpected={len(unexpected)})")
    assert not unexpected, f"unexpected encoder keys: {unexpected[:5]}"
    del src
```
(`restore_from` for a local `.nemo`; the round-trip was proven in Task 0.)

- [ ] **Step 3: Verify it loads on the smoke checkpoint**

Run:
```bash
HF_HUB_OFFLINE=1 rasr train run -c configs/train/rtx6kpro/parakeet-higgs-smoke.yaml \
  model.init_encoder_from=ckpt/ssl-smoke/final.nemo \
  trainer.max_steps=20 output.dir=ckpt/ft-from-ssl-smoke 2>&1 | tail -20
```
Expected: prints `loaded SSL encoder ... (missing=0 unexpected=0)` then trains 20 steps.

- [ ] **Step 4: Commit**

```bash
git add src/rasr/train/config.py src/rasr/train/nemo.py
git commit -m "feat(train): finetune can init encoder from an SSL .nemo"
```

### Task 8: Final SSL recipe wiring + run plan

**Files:**
- Create: `configs/train/rtx6kpro/parakeet-higgs-ssl.yaml`

- [ ] **Step 1: Write the SSL-then-finetune recipe**

Clone `parakeet-radiotalk`-style finetune (200k clips, 20k steps) but with the
tuned HPs and the SSL encoder:
```yaml
defaults: [base, rtx6kpro/hw]
name: parakeet-higgs-ssl
model:
  scheme: parakeet
  ref: nvidia/parakeet-tdt-0.6b-v3
  init_encoder_from: ckpt/parakeet-ssl-tartan/final.nemo
optimizer:
  lr: 2.5e-5            # from docs/higgs-lr-warmup-sweep.md
scheduler:
  warmup_steps: 500
data:
  train:
    - dataset: hf:twangodev/radiotalk-us-audio-higgs-noisy:train
      weight: 1.0
      limit: 200000
  validation:
    - dataset: hf:jlvdoorn/atco2-asr:validation
trainer:
  max_steps: 20000
  val_check_interval: 2000
output:
  dir: ckpt/${name}
```

- [ ] **Step 2: Dry-run validate**

Run: `HF_HUB_OFFLINE=1 rasr train run -c configs/train/rtx6kpro/parakeet-higgs-ssl.yaml --dry-run`
Expected: config summary loads (lr 2.5e-5, warmup 500, init_encoder_from set).

- [ ] **Step 3: Commit**

```bash
git add configs/train/rtx6kpro/parakeet-higgs-ssl.yaml
git commit -m "feat(train): SSL->finetune recipe (Tartan CPT encoder + tuned Higgs HPs)"
```

- [ ] **Step 4: Document the full run order (no execution here)**

Append to `docs/superpowers/specs/2026-05-26-ssl-continued-pretraining-design.md`
a short "Run order" section: (1) `rasr train ssl -c parakeet-ssl-tartan.yaml`
(overnight), (2) `rasr train run -c parakeet-higgs-ssl.yaml`, (3) eval both
`parakeet-higgs-ssl` and the SSL-free baseline on ATCO2 and compare WER.

```bash
git add docs/superpowers/specs/2026-05-26-ssl-continued-pretraining-design.md
git commit -m "docs: SSL run order"
```

---

## Self-review notes

- **Spec coverage:** Stage 0 spike (risk), Stage 1 (Tasks 1-3, data prep),
  Stage 2 (Tasks 4-6, CPT), Stage 3 (Tasks 7-8, transfer + finetune), final
  comparison documented in Task 8. All spec sections mapped.
- **Known uncertainty, deliberately deferred to Task 0:** the exact NeMo SSL
  model class and config keys. Tasks 4-5 explicitly say "use the class/keys
  discovered in Task 0" rather than inventing them — Task 0 is a hard gate.
- **Type consistency:** `SSLDatasetCfg`/`SSLBlockCfg`/`SSLTrainConfig`,
  `build_ssl_manifest`, `model.init_encoder_from`, and the `ssl` CLI command are
  referenced consistently across tasks.
