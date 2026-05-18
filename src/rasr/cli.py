from __future__ import annotations

import typer

from rasr.eval.cli import app as eval_app
from rasr.train.cli import app as train_app

app = typer.Typer(
    name="rasr",
    help="rasr — finetuning toolkit for ASR on ATC audio.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval", help="Evaluate ASR models against datasets.")
app.add_typer(train_app, name="train", help="Finetune ASR models.")


if __name__ == "__main__":
    app()
