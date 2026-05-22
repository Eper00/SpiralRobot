import typer
import yaml
import torch

from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


app = typer.Typer()


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@app.command()
def train(config_path: str):

    # -------------------------
    # CONFIG
    # -------------------------
    cfg = load_config(config_path)

    env_cfg = cfg["env"]
    rl_cfg = cfg["rl"]
    policy_cfg = cfg["policy"]
    warm_start = env_cfg["warm_start"]
    env_id = env_cfg["env_id"]
    n_train_envs = env_cfg["n_training_envs"]
    n_eval_envs = env_cfg["n_eval_envs"]
    seed = env_cfg["seed"]

    total_timesteps = rl_cfg["total_timesteps"]
    eval_freq = rl_cfg["eval_freq"]
    n_eval_episodes = rl_cfg["n_eval_episodes"]

    net_arch = policy_cfg["net_arch"]

    # -------------------------
    # RUN DIRS
    # -------------------------
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_rl"+f"_{env_id}")
    root = Path("results") / run_name
    model_dir = root / "models"
    log_dir = root / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # ENVS
    # -------------------------
    train_env = make_vec_env(
        env_id,
        n_envs=n_train_envs,
        seed=seed
    )

    eval_env = make_vec_env(
        env_id,
        n_envs=n_eval_envs,
        seed=seed
    )

    # -------------------------
    # EVAL CALLBACK
    # -------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    # -------------------------
    # PPO MODEL
    # -------------------------
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=dict(
            net_arch=net_arch,
        ),
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=seed,
    )

    # -------------------------
    # LOAD BC WEIGHTS (optional warm start)
    # -------------------------
   
    bc_path = Path(cfg["rl"]["bc_path"])

    if warm_start and bc_path.exists():
        state_dict = torch.load(bc_path, map_location="cpu")
        model.policy.load_state_dict(state_dict, strict=False)
        print(f"Loaded BC from {bc_path}")
    else:
        print("No BC weights found. Training from scratch.")

    # -------------------------
    # TRAIN PPO
    # -------------------------
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

    # -------------------------
    # SAVE FINAL MODEL
    # -------------------------
    model.save(str(model_dir / "final_model"))

    print(f"Done. Results saved to: {root}")


if __name__ == "__main__":
    app()