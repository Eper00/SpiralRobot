"""Clean RL evaluation script (CLI version, same style as training)."""

import time
from pathlib import Path
from typing import Optional

import yaml
import typer
import numpy as np
from stable_baselines3 import PPO
from rich.console import Console

from rl.environment import TentacleTargetFollowingEnv
from common.loaders import  RLTrainingConfig

console = Console()
app = typer.Typer()


# ----------------------------
# CONFIG LOADER
# ----------------------------
def load_config(path: Optional[str]) -> RLTrainingConfig:
    if path is None:
        return RLTrainingConfig()

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = raw.get("rl_training", raw)
    return RLTrainingConfig.model_validate(cfg)


# ----------------------------
# EVALUATE COMMAND
# ----------------------------
@app.command()
def evaluate(
    model_path: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    num_episodes: Optional[int] = typer.Option(None),
    render: bool = typer.Option(True),
    deterministic: Optional[bool] = typer.Option(None),
    render_delay: Optional[float] = typer.Option(None),
    save_results: bool = typer.Option(False),
    verbose: bool = typer.Option(True),
):
    # --------------------
    # CONFIG
    # --------------------
    cfg = load_config(config)

    if num_episodes is not None:
        cfg.rl_evaluation.num_episodes = num_episodes
    if deterministic is not None:
        cfg.rl_evaluation.deterministic_actions = deterministic
    if render_delay is not None:
        cfg.rl_evaluation.render_delay = render_delay

    cfg.rl_evaluation.render_mode = "human" if render else None

    # --------------------
    # MODEL
    # --------------------
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = PPO.load(str(model_path))

    # --------------------
    # ENV
    # --------------------
    env = TentacleTargetFollowingEnv(
        config=cfg.rl_env,
        render_mode=cfg.rl_evaluation.render_mode,
    )

    if verbose:
        console.print(f"Episodes: {cfg.rl_evaluation.num_episodes}")
        console.print(f"Render: {cfg.rl_evaluation.render_mode}")
        console.print(f"Deterministic: {cfg.rl_evaluation.deterministic_actions}")

    # --------------------
    # METRICS
    # --------------------
    rewards, lengths, distances = [], [], []
    success = 0
    threshold = 0.5

    try:
        for ep in range(cfg.rl_evaluation.num_episodes):

            obs, _ = env.reset()
            done = False

            ep_reward = 0.0
            ep_len = 0

            while not done:

                action, _ = model.predict(
                    obs,
                    deterministic=cfg.rl_evaluation.deterministic_actions,
                )

                obs, reward, terminated, truncated, info = env.step(action)
                
                done = terminated or truncated
                ep_reward += reward
                ep_len += 1

                if cfg.rl_evaluation.render_mode == "human":
                    env.render()
                    if cfg.rl_evaluation.render_delay:
                        time.sleep(cfg.rl_evaluation.render_delay)

            final_dist = info.get("distance_to_target", float("inf"))

            rewards.append(ep_reward)
            lengths.append(ep_len)
            distances.append(final_dist)

            if final_dist <= threshold:
                success += 1

            if verbose:
                console.print(f"Ep {ep}: R={ep_reward:.3f}, L={ep_len}")

    finally:
        env.close()

    # --------------------
    # RESULTS
    # --------------------
    if not rewards:
        console.print("No episodes run.")
        return

    console.print("\n=== RESULTS ===")
    console.print(f"Episodes: {len(rewards)}")
    console.print(f"Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    console.print(f"Length: {np.mean(lengths):.1f}")
    console.print(f"Distance: {np.mean(distances):.4f}")
    console.print(f"Success: {100 * success / len(rewards):.1f}%")

    if save_results:
        out = model_path.parent / f"eval_{model_path.stem}.txt"
        out.write_text(
            f"Reward mean: {np.mean(rewards)}\n"
            f"Length mean: {np.mean(lengths)}\n"
            f"Distance mean: {np.mean(distances)}\n"
        )
        console.print(f"Saved: {out}")

# ----------------------------
# ENTRYPOINT
# ----------------------------
if __name__ == "__main__":
    app()