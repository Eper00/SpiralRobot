"""Behavior cloning trajectory evaluation + visualization."""
import mujoco
import time
from pathlib import Path
from typing import Optional

import typer
import yaml
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from rich.console import Console

from imitation.algorithms import bc
from imitation.data.types import Trajectory

from il.environment import TentacleTargetFollowingIL
from common.loaders import RLTrainingConfig

from common.support import (
    _get_tip_position,
    _action_to_ctrl,
    _read_dataset
)

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
    model_path: str = typer.Option(..., "--model"),
    num_trajectories: int = typer.Option(5),
    visualize: bool = typer.Option(True),
):

    # -----------------------------------------------------
    # config
    # -----------------------------------------------------
    cfg = load_config(config)

    console.print("\n=== BC POLICY EVALUATION ===")

    # -----------------------------------------------------
    # env
    # -----------------------------------------------------
    env = TentacleTargetFollowingIL(
        config=cfg.rl_env
    )

    # -----------------------------------------------------
    # dataset (for BC init compatibility)
    # -----------------------------------------------------
    rng = np.random.default_rng(0)

    states, actions = _read_dataset(
        "/home/tomi/SpiralRobot/demonstration_dataset.npz"
    )

    demonstrations = []

    for i in range(len(states)):

        demonstrations.append(
            Trajectory(
                obs=states[i],
                acts=actions[i],
                infos=np.array(
                    [{} for _ in range(len(actions[i]))],
                    dtype=object
                ),
                terminal=True,
            )
        )

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=demonstrations,
        rng=rng,
        device="cuda" if th.cuda.is_available() else "cpu",
    )

    # -----------------------------------------------------
    # load model
    # -----------------------------------------------------
    console.print(f"Loading policy: {model_path}")

    bc_trainer.policy.load_state_dict(
        th.load(model_path, map_location=bc_trainer.policy.device)
    )

    policy = bc_trainer.policy
    policy.eval()

    # -----------------------------------------------------
    # metrics
    # -----------------------------------------------------
    final_distances = []

    # -----------------------------------------------------
    # rollout evaluation
    # -----------------------------------------------------
    for traj_idx in range(num_trajectories):

        obs= env.reset()
        trajectory = []

        for step in range(env._max_episode_steps):
            # -----------------------------------------
            # obs → tensor
            # -----------------------------------------
            obs_tensor = th.tensor(
                obs,
                dtype=th.float32,
                device=bc_trainer.policy.device
            ).unsqueeze(0)

            # -----------------------------------------
            # forward pass
            # -----------------------------------------
            with th.no_grad():

                action_tensor = policy(obs_tensor)

                if isinstance(action_tensor, tuple):
                    action_tensor = action_tensor[0]

            action = action_tensor.cpu().numpy()[0]
            action = np.clip(action, -1, 1)

            ctrl = _action_to_ctrl(
                action,
                env.actuator_low,
                env.actuator_high
            )

            # -----------------------------------------
            # simulate
            # -----------------------------------------
            for _ in range(env.frame_skip):
                env.data.ctrl[:] = ctrl
                mujoco.mj_step(env.model, env.data)

            obs = env._get_obs()

            tip = _get_tip_position(env.model, env.data)
            trajectory.append(tip.copy())

        # -------------------------------------------------
        # metrics
        # -------------------------------------------------
        trajectory = np.array(trajectory)

        final_tip = trajectory[-1]
        target = env.target_position

        final_distance = np.linalg.norm(final_tip - target)

        final_distances.append(final_distance)

        console.print(
            f"[Trajectory {traj_idx}] "
            f"Final distance: {final_distance:.4f}"
        )

    # -----------------------------------------------------
    # summary
    # -----------------------------------------------------
    final_distances = np.array(final_distances)

    console.print("\n=== SUMMARY ===")
    console.print(f"Mean final distance: {final_distances.mean():.4f}")
    console.print(f"Std final distance:  {final_distances.std():.4f}")
    console.print(f"Min final distance:  {final_distances.min():.4f}")
    console.print(f"Max final distance:  {final_distances.max():.4f}")
# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
if __name__ == "__main__":
    app()