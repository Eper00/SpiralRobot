"""Behavior Cloning training script."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import typer
import yaml
import numpy as np
import torch as th

from imitation.algorithms import bc

from rich.console import Console

from il.environment import TentacleTargetFollowingIL
from common.loaders import RLEnvironmentConfig

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()


# ---------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------
def load_config(path: Optional[str]) -> dict:
    if path is None:
        return {}

    with open(path, "r") as f:
        return yaml.safe_load(f)["rl_training"]


# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------
@app.command()
def train(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    # ------------------------------------------------
    # config
    # ------------------------------------------------
    cfg = load_config(config)

    env_cfg = cfg["rl_env"]
    train_cfg = cfg["rl_training_params"]
    ir_cfg=cfg["ir"]
    # ------------------------------------------------
    # dirs
    # ------------------------------------------------
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_il")
    root = Path("results") / run_name
    model_dir = root / "models"
    log_dir = root / "logs" 
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # env
    # ------------------------------------------------
    env = TentacleTargetFollowingIL(
        config=RLEnvironmentConfig(**env_cfg)
    )

    # ------------------------------------------------
    # generate demonstrations (FIXED)
    # ------------------------------------------------
    console.print("Generating demonstrations...")

    trajectories = env.generate_demonstrations()
    env.save_demonstration_dataset()

    if len(trajectories) == 0:
        raise ValueError("No trajectories generated!")

    console.print(f"Trajectories: {len(trajectories)}")

    # ------------------------------------------------
    # BC trainer
    # ------------------------------------------------
    rng = np.random.default_rng(0)
    train_cfg = cfg["ir"]
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=trajectories,
        batch_size=train_cfg["BC_batch_size"],
        rng=rng,
        device="cuda" if th.cuda.is_available() else "cpu",
        
    )

    # ------------------------------------------------
    # train
    # ------------------------------------------------
    console.print("Training BC policy...")

    bc_trainer.train(
        n_epochs=train_cfg["BC_epochs"],
        progress_bar=True,
    )

    # ------------------------------------------------
    # save
    # ------------------------------------------------
    save_path = model_dir / "bc_policy.pt"

    th.save(
        bc_trainer.policy.state_dict(),
        save_path,
    )

    console.print(f"Saved BC policy to: {save_path}")


# ---------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------
if __name__ == "__main__":
    app()