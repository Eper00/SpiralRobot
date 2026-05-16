import typer
import yaml
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
    # load config
    # -------------------------
    cfg = load_config(config_path)

    env_cfg = cfg["env"]
    train_cfg = cfg["train"]

    env_id = env_cfg["env_id"]
    n_training_envs = env_cfg["n_training_envs"]
    n_eval_envs = env_cfg["n_eval_envs"]
    seed = env_cfg["seed"]

    total_timesteps = train_cfg["total_timesteps"]
    eval_freq = train_cfg["eval_freq"]
    n_eval_episodes = train_cfg["n_eval_episodes"]

    # -------------------------
    # run folders
    # -------------------------
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_rl")
    root = Path("results") / run_name
    model_dir = root / "models"
    log_dir = root / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # envs
    # -------------------------
    train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=seed)

    eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=seed)

    # -------------------------
    # eval callback
    # -------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir),
        eval_freq=max(500 // n_training_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # -------------------------
    # PPO model
    # -------------------------
    model = PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=seed,
    )

    # -------------------------
    # training
    # -------------------------
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

    # -------------------------
    # save final model
    # -------------------------
    model.save(str(model_dir / "final_model"))

    print(f"Done. Results saved to: {root}")


if __name__ == "__main__":
    app()