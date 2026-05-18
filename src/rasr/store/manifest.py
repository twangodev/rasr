from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class RunManifest:
    run_id: str
    started_at: str
    model_id: str
    dataset_id: str
    normalize: dict
    git_sha: str | None
    n_utterances: int


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def new_run_id(model_id: str, dataset_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha1(f"{model_id}|{dataset_id}|{stamp}".encode()).hexdigest()[:8]
    return f"{stamp}-{h}"


def write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
