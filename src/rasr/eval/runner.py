from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from radiotalk.normalize import normalize

from rasr.datasets import load_dataset
from rasr.eval.metrics.wer import wer
from rasr.models import load_model
from rasr.store import RunManifest, git_sha, new_run_id, write_manifest


@dataclass(slots=True)
class EvalResult:
    run_dir: Path
    wer: float


def _radiotalk_version() -> str:
    try:
        return version("radiotalk")
    except PackageNotFoundError:
        return "unknown"


def run_eval(
    model_id: str,
    dataset_path: Path,
    out_dir: Path,
    limit: int | None = None,
) -> EvalResult:
    dataset = load_dataset(dataset_path)
    model = load_model(model_id)

    run_id = new_run_id(model_id, dataset.id)
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    transcripts_path = run_dir / "transcripts.jsonl"
    hyps: list[str] = []
    refs: list[str] = []

    with transcripts_path.open("w") as f:
        for i, utt in enumerate(dataset):
            if limit is not None and i >= limit:
                break
            hyp_raw = model.transcribe(utt.audio, utt.sr)
            hyp = normalize(hyp_raw)
            ref = normalize(utt.reference)
            hyps.append(hyp)
            refs.append(ref)
            f.write(
                json.dumps(
                    {
                        "utt_id": utt.utt_id,
                        "reference_raw": utt.reference,
                        "reference": ref,
                        "hypothesis_raw": hyp_raw,
                        "hypothesis": hyp,
                    }
                )
                + "\n"
            )

    score = wer(hyps, refs) if hyps else float("nan")

    manifest = RunManifest(
        run_id=run_id,
        started_at=run_id.split("-")[0],
        model_id=model_id,
        dataset_id=dataset.id,
        normalize={
            "source": "radiotalk",
            "version": _radiotalk_version(),
            "pipeline": "default",
        },
        git_sha=git_sha(),
        n_utterances=len(hyps),
    )
    write_manifest(run_dir, manifest)
    (run_dir / "scores.json").write_text(json.dumps({"wer": score}, indent=2))

    return EvalResult(run_dir=run_dir, wer=score)
