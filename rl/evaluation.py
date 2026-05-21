import typer
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from rl.training import load_config
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
app = typer.Typer()
@app.command()
def visualize(config_path: str, model_path: str):
    cfg = load_config(config_path)

    env_id = cfg["env"]["env_id"]
    n_envs = cfg["env"]["n_eval_envs"]
    vec_env = make_vec_env(env_id, n_envs=1)

    model = PPO.load(model_path)
    print(model.policy)
    obs = vec_env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

@app.command()
def eval(
    config_path: str,
    model_path: str,
    render: bool = True,  
):

    cfg = load_config(config_path)

    env_id = cfg["env"]["env_id"]
    seed = cfg["env"]["seed"]
    n_episodes = cfg["train"]["n_eval_episodes"]
    n_eval_envs = cfg["env"]["n_eval_envs"]
    vec_env = make_vec_env(env_id, n_envs=n_eval_envs,seed=seed)
    model = PPO.load(model_path)
    
   

    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=n_episodes,
        deterministic=True
    )
    vec_env.close()

    print("\n===== NUMERICAL EVAL =====")
    print("Mean reward:", mean_reward)
    print("Std reward :", std_reward)


if __name__ == "__main__":
    app()