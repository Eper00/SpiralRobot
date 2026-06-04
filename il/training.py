import torch

from il.expert import TentacleTargetFollowingExpert
from toy.rl.training import load_config
import typer
from rich.console import Console
import logging

from typing import Optional
from imitation.algorithms.bc import BC
from stable_baselines3.common.policies import ActorCriticPolicy
from pathlib import Path
from datetime import datetime
from imitation.util import logger as imit_logger
import torch
app = typer.Typer()
logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()


@app.command()
def train(config: Optional[str] = typer.Option(None, "--config", "-c")):
    cfg = load_config(config)
    expert = TentacleTargetFollowingExpert(cfg)
    print("Generating demonstrations...")
    

    trajectories = expert.generate_demonstrations()

    policy = ActorCriticPolicy(
        observation_space=expert.observation_space,
        action_space=expert.action_space,
        lr_schedule=lambda _: expert.lr,
        net_arch=expert.net_arch,
        activation_fn=expert.activation_fn
        )
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_bc")
    root = Path("results") / run_name
    model_dir = root / "models"
    log_dir = root / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = imit_logger.configure(
    folder=str(log_dir),
    format_strs=["tensorboard", "stdout"],
    )
    bc_trainer = BC(
            observation_space=expert.observation_space,
            action_space=expert.action_space,
            rng=expert.seed,
            demonstrations=trajectories,
            policy=policy,
            custom_logger=logger
        )

    bc_trainer.train(n_epochs=expert.bc_epochs)
    torch.save(
    bc_trainer.policy.state_dict(),
    model_dir / "bc_policy.pt"
    )

    print(f"Done. Results saved to: {root}")
if __name__ == "__main__":
    app()