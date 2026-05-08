"""Behavior cloning trajectory evaluation + visualization."""

import time
from pathlib import Path
from typing import Optional

import typer
import yaml
import numpy as np
from rich.console import Console

from ir.expert import TentacleTargetFollowingClone
from common.loaders import RLTrainingConfig


console = Console()
app = typer.Typer()


# ---------------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------------
def load_config(path: Optional[str]) -> RLTrainingConfig:

    if path is None:
        return RLTrainingConfig()

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = raw.get("rl_training", raw)

    return RLTrainingConfig.model_validate(cfg)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------
@app.command()
def evaluate(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    num_trajectories: int = typer.Option(10),
    save_dataset: bool = typer.Option(False),
    dataset_path: str = typer.Option("bc_dataset.npz"),
    visualize: bool = typer.Option(True),
):

    # -----------------------------------------------------
    # config
    # -----------------------------------------------------
    cfg = load_config(config)
    # overwrite demo count
    cfg.ir.demostraion_number = num_trajectories

    console.print("\n=== Behavior Cloning Evaluation ===")
    console.print(f"Trajectories: {num_trajectories}")

    # -----------------------------------------------------
    # clone env
    # -----------------------------------------------------
    clone = TentacleTargetFollowingClone(
        config=cfg.rl_env
    )

    all_states = []
    all_actions = []

    trajectory_lengths = []
    workspace_points = []

    # -----------------------------------------------------
    # generate trajectories
    # -----------------------------------------------------
    for i in range(num_trajectories):

        states =clone.demonstration_states
        actions=clone.demonstration_actions
        all_states.append(states)
        all_actions.append(actions)

        trajectory_lengths.append(len(states))

        workspace_points.append(states)

        console.print(
            f"[{i+1}/{num_trajectories}] "
            f"steps={len(states)} "
            f"start={states[0]} "
            f"end={states[-1]}"
        )

    # -----------------------------------------------------
    # concatenate
    # -----------------------------------------------------
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    workspace_points = np.concatenate(workspace_points, axis=0)

    # -----------------------------------------------------
    # metrics
    # -----------------------------------------------------
    mins = workspace_points.min(axis=0)
    maxs = workspace_points.max(axis=0)
    means = workspace_points.mean(axis=0)

    console.print("\n=== DATASET STATS ===")
    console.print(f"Total samples: {len(all_states)}")
    console.print(f"Mean trajectory length: {np.mean(trajectory_lengths):.2f}")

    console.print("\nWorkspace:")
    console.print(f"X: {mins[0]:.4f} -> {maxs[0]:.4f}")
    console.print(f"Y: {mins[1]:.4f} -> {maxs[1]:.4f}")
    console.print(f"Z: {mins[2]:.4f} -> {maxs[2]:.4f}")

    console.print("\nMean position:")
    console.print(means)

    # -----------------------------------------------------
    # save dataset
    # -----------------------------------------------------
    if save_dataset:

        np.savez_compressed(
            dataset_path,
            states=all_states,
            actions=all_actions,
        )

        console.print(f"\nSaved dataset: {dataset_path}")

 

# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
if __name__ == "__main__":
    app()