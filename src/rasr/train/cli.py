from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from rasr.train.config import load_train_config

app = typer.Typer(no_args_is_help=True, help="Finetune ASR models.")
console = Console()


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to a training config YAML under configs/train/.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    overrides: Annotated[
        list[str] | None,
        typer.Argument(
            help="Hydra-style overrides, e.g. trainer.max_steps=100000",
        ),
    ] = None,
) -> None:
    """Run a training job from a YAML recipe."""
    cfg = load_train_config(config, overrides=overrides)
    console.print(f"[green]Loaded config:[/green] {cfg.name}")
    console.print(f"  model:    {cfg.model.scheme}:{cfg.model.ref}")
    console.print(f"  train:    {len(cfg.data.train)} dataset(s)")
    for ds in cfg.data.train:
        limit_s = f" (limit={ds.limit})" if ds.limit else ""
        console.print(f"            - {ds.dataset}  weight={ds.weight}{limit_s}")
    console.print(f"  val:      {len(cfg.data.validation)} dataset(s)")
    for ds in cfg.data.validation:
        console.print(f"            - {ds.dataset}")
    console.print(f"  steps:    {cfg.trainer.max_steps} (val every {cfg.trainer.val_check_interval})")
    console.print(f"  batch:    {cfg.trainer.batch_size} × {cfg.trainer.devices} GPU")
    console.print(f"  precision: {cfg.trainer.precision}")
    console.print(f"  output:   {cfg.output.dir}")
    console.print()
    console.print(
        "[yellow]Trainer backend not yet wired — "
        "this command currently validates the config only.[/yellow]"
    )
