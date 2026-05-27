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
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Load and validate the config but don't kick off training.",
        ),
    ] = False,
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
    console.print(
        f"  steps:    {cfg.trainer.max_steps} "
        f"(val every {cfg.trainer.val_check_interval})"
    )
    console.print(
        f"  batch:    {cfg.trainer.batch_size} x {cfg.trainer.devices} GPU"
    )
    console.print(f"  precision: {cfg.trainer.precision}")
    console.print(f"  output:   {cfg.output.dir}")

    if dry_run:
        console.print("[yellow]--dry-run: skipping trainer.[/yellow]")
        return

    from rasr.train.nemo import run as nemo_run

    final_path = nemo_run(cfg)
    console.print(f"[green]Training done.[/green] Saved: {final_path}")


@app.command()
def ssl(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    overrides: Annotated[list[str] | None, typer.Argument()] = None,
) -> None:
    """Continue-pretrain an encoder with SSL on unlabeled audio."""
    from rasr.train.config import load_ssl_train_config

    cfg = load_ssl_train_config(config, overrides=overrides)
    console.print(f"[green]Loaded SSL config:[/green] {cfg.name}")
    console.print(f"  source:   {cfg.ssl.source}")
    console.print(f"  init_from:{cfg.ssl.init_encoder_from}")
    console.print(f"  steps:    {cfg.trainer.max_steps}")
    console.print(f"  output:   {cfg.output.dir}")
    if dry_run:
        console.print("[yellow]--dry-run: skipping trainer.[/yellow]")
        return
    from rasr.train.ssl import run as ssl_run

    console.print(f"[green]SSL done.[/green] Saved: {ssl_run(cfg)}")
