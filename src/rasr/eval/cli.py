from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

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
        str,
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset spec: 'hf:<owner>/<repo>[:<split>]' or path to a JSONL manifest.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Runs directory."),
    ] = Path("runs"),
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Utterances per model call."),
    ] = 8,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Limit number of utterances."),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="ISO 639-1 language hint (e.g. 'en'). Unset = auto-detect.",
        ),
    ] = None,
) -> None:
    """Transcribe a dataset with a model and score WER."""
    result = run_eval(
        model_id=model,
        dataset_spec=dataset,
        out_dir=out,
        batch_size=batch_size,
        limit=limit,
        language=language,
    )
    console.print(f"[green]Run complete:[/green] {result.run_dir}")

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    for key, value in result.summary.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    console.print(table)
