from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from itertools import batched, islice
from pathlib import Path

from radiotalk.normalize import normalize

from rasr.datasets import load_dataset
from rasr.eval.metrics import corpus_wer, summarize, utt_wer
from rasr.models import load_model
from rasr.models.base import AudioInput
from rasr.store import RunManifest, git_sha, new_run_id, write_manifest


@dataclass(slots=True)
class EvalResult:
    run_dir: Path
    corpus_wer: float
    summary: dict


def _radiotalk_version() -> str:
    try:
        return version("radiotalk")
    except PackageNotFoundError:
        return "unknown"


def run_eval(
    model_id: str,
    dataset_spec: str,
    out_dir: Path,
    batch_size: int = 8,
    limit: int | None = None,
    language: str | None = None,
) -> EvalResult:
    dataset = load_dataset(dataset_spec)
    model = load_model(model_id, language=language)

    run_id = new_run_id(model_id, dataset.id)
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    transcripts_path = run_dir / "transcripts.jsonl"
    hyps: list[str] = []
    refs: list[str] = []
    per_utt: list[float] = []

    source = iter(dataset)
    if limit is not None:
        source = islice(source, limit)

    with transcripts_path.open("w") as f:
        for batch in batched(source, batch_size):
            audios = [AudioInput(u.audio, u.sr) for u in batch]
            hyps_raw = model.transcribe(audios)
            for utt, hyp_raw in zip(batch, hyps_raw, strict=True):
                hyp = normalize(hyp_raw)
                ref = normalize(utt.reference)
                w = utt_wer(ref, hyp)
                hyps.append(hyp)
                refs.append(ref)
                per_utt.append(w)
                f.write(
                    json.dumps(
                        {
                            "utt_id": utt.utt_id,
                            "reference_raw": utt.reference,
                            "reference": ref,
                            "hypothesis_raw": hyp_raw,
                            "hypothesis": hyp,
                            "wer": w,
                        }
                    )
                    + "\n"
                )

    corpus = corpus_wer(refs, hyps)
    summary = {"corpus_wer": corpus, **summarize(per_utt)}

    manifest = RunManifest(
        run_id=run_id,
        started_at=run_id.split("-")[0],
        model_id=model_id,
        dataset_id=dataset.id,
        normalize={
            "source": "radiotalk",
            "version": _radiotalk_version(),
            "pipeline": "default",
            "eval_transform": "lowercase+strip-punct",
        },
        git_sha=git_sha(),
        n_utterances=len(hyps),
    )
    write_manifest(run_dir, manifest)
    (run_dir / "scores.json").write_text(json.dumps(summary, indent=2))

    return EvalResult(run_dir=run_dir, corpus_wer=corpus, summary=summary)
