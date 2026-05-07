"""RL training configuration (standalone, no BaseConfig dependency)."""

from typing import List, Optional
from pydantic import BaseModel, Field
import yaml


# -------------------------
# ENV CONFIG
# -------------------------
class RLEnvironmentConfig(BaseModel):
    xml_file: str = Field(default="assets/simulation/tentacle.xml")
    simulation_length_seconds: float = Field(default=4.0)
    time_between_steps_seconds: float = Field(default=0.04)

    initial_actuator_position: List[float] = Field(default=[0.147, 0.25])

    reward_distance_scale: float = Field(default=100.0)
    action_change_penalty_scale: float = Field(default=0.25)
    control_penalty_scale: float = Field(default=0.1)

    tip_site_name: str = Field(default="tip_center")
    target_bounds_min: List[float] = Field(default=[-0.2, -0.2, 0.125])
    target_bounds_max: List[float] = Field(default=[0.2, 0.2, 0.1255])

    num_frames: int = Field(default=6)
    include_actuator_lengths_in_obs: bool = Field(default=True)

    fixed_target_position: Optional[List[float]] = None


# -------------------------
# TRAINING PARAMS
# -------------------------
class RLTrainingParamsConfig(BaseModel):
    num_envs: int = 20
    total_timesteps: int = 5_000_000

    eval_freq: int = 10_000
    n_eval_episodes: int = 10

    save_freq: int = 50_000
    log_dir_base: str = "results"

    learning_rate: float = 1e-4
    n_steps: int = 4096
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    target_kl: float = 0.08

    net_arch: str = "256-128"
    activation_fn: str = "Tanh"


# -------------------------
# EVAL CONFIG
# -------------------------
class RLEvaluationConfig(BaseModel):
    num_episodes: int = 1
    render_delay: float = 0.05
    deterministic_actions: bool = True
    render_mode: Optional[str] = None
    grid_size: int = 20
    space_limit: float = 0.2


# -------------------------
# ROOT CONFIG
# -------------------------
class RLTrainingConfig(BaseModel):
    rl_env: RLEnvironmentConfig = Field(default_factory=RLEnvironmentConfig)
    rl_training_params: RLTrainingParamsConfig = Field(default_factory=RLTrainingParamsConfig)
    rl_evaluation: RLEvaluationConfig = Field(default_factory=RLEvaluationConfig)


# -------------------------
# YAML LOADER
# -------------------------
def load_rl_config(path: str) -> RLTrainingConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)["rl_training"]

    return RLTrainingConfig.model_validate(raw)