from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from rasr.eval.runner import run_eval

app = typer.Typer(no_args_is_help=True, help="Evaluate ASR models against datasets.")
console = Console()


@app.command()
def run(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model ID, e.g. 'whisper:openai/whisper-tiny'.",
        ),
    ],
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to dataset manifest (JSONL).",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Runs directory."),
    ] = Path("runs"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Limit number of utterances."),
    ] = None,
) -> None:
    """Transcribe a dataset with a model and score WER."""
    result = run_eval(
        model_id=model,
        dataset_path=dataset,
        out_dir=out,
        limit=limit,
    )
    console.print(f"[green]Run complete:[/green] {result.run_dir}")
    console.print(f"[bold]WER:[/bold] {result.wer:.4f}")
