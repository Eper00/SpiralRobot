import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
import typer
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.util import logger as imit_logger
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
   
    env_id = env_cfg["env_id"]
    policy_cfg = cfg["policy"]
    seed = env_cfg["seed"]
    n_envs = env_cfg["n_training_envs"]
    seed = env_cfg["seed"]
    bc_cfg = cfg["bc"]
    n_expert_episodes = bc_cfg["n_expert_episodes"]
    n_epochs = bc_cfg["n_epochs"]
    n_eval_envs = env_cfg["n_eval_envs"]
   
    net_arch = policy_cfg["net_arch"]
    lr = float(policy_cfg["learning_rate"])
    activation_fn = policy_cfg["activation_fn"]
    if activation_fn == "relu":
        activation_fn = torch.nn.ReLU
    elif activation_fn == "tanh":
        activation_fn = torch.nn.Tanh
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")
    # -------------------------
    # run folders
    # -------------------------
    rng = np.random.default_rng(seed)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_bc"+f"_{env_id}")
    root = Path("results") / run_name
    model_dir = root / "models"
    log_dir = root / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # env
    # -------------------------
    env = make_vec_env(
        "seals:seals/"+env_id,
        n_envs=n_envs,
        rng=rng,
        post_wrappers=[
            lambda env, _: RolloutInfoWrapper(env)
        ]
    )

    # -------------------------
    # load expert
    # -------------------------
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-"+env_id,
        venv=env,
    )

    # -------------------------
    # collect demonstrations
    # -------------------------
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(
            min_timesteps=None,
            min_episodes=n_expert_episodes
        ),
        rng=rng,
    )

    transitions = rollout.flatten_trajectories(
        rollouts
    )

    # -------------------------
    # BC trainer
    # -------------------------

    logger = imit_logger.configure(
        folder=str(log_dir),
        format_strs=["tensorboard", "stdout"],
    )
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: lr,
        net_arch=net_arch,
        activation_fn=activation_fn
    )
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        policy=policy,  
        custom_logger=logger

 
    )

    # -------------------------
    # train BC
    # -------------------------
    bc_trainer.train(
        n_epochs=n_epochs,
        log_rollouts_venv=env,
        log_rollouts_n_episodes=5,
        
    )

    # -------------------------
    # evaluate
    # -------------------------
        # -------------------------
    # eval callback
    # -------------------------

    reward, _ = evaluate_policy(
        bc_trainer.policy,
        env,
        n_eval_episodes=n_eval_envs
    )

    print(f"Reward: {reward}")

    # -------------------------
    # save weights
    # -------------------------
    torch.save(
        bc_trainer.policy.state_dict(),
        model_dir / "bc_policy.pt"
    )

    print(f"Done. Results saved to: {root}")


if __name__ == "__main__":
    app()