"""MuJoCo-based reinforcement learning environment for tentacle robots."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import mujoco.viewer
import os
from collections import deque
from loaders import RLEnvironmentConfig
from typing import  Optional, Dict, Any
import time
class TentacleTargetFollowingEnv(gym.Env):


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config: Optional[RLEnvironmentConfig] = None,
        render_mode: str = None,
    ):
        super().__init__()

        self.config = config
        self.render_mode = render_mode
        xml_file = "../assets/simulation/tentacle.xml"
        if not os.path.exists(xml_file):
            script_dir = os.path.dirname(__file__)
            abs_xml_file = os.path.join(script_dir, xml_file)
            if not os.path.exists(abs_xml_file):
                raise FileNotFoundError(
                    f"Cannot find MuJoCo XML file at {xml_file} or {abs_xml_file}"
                )
            xml_file = abs_xml_file
        
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        # Time-based parameters from config
        self.simulation_length_seconds = self.config.simulation_length_seconds
        self.time_between_steps_seconds = self.config.time_between_steps_seconds
        self.timestep = self.model.opt.timestep

        # Calculate frame_skip based on desired time between steps
        self.frame_skip = max(1, round(self.time_between_steps_seconds / self.timestep))
        self.time_per_step = self.frame_skip * self.timestep

        # Calculate max_episode_steps based on simulation length and actual time per step
        self._max_episode_steps = int(
            self.simulation_length_seconds / self.time_per_step
        )

        # Extract config parameters
        self.initial_actuator_config = self.config.initial_actuator_position

        self.tip_site_name = self.config.tip_site_name
        self.target_bounds_min = np.array(self.config.target_bounds_min)
        self.target_bounds_max = np.array(self.config.target_bounds_max)


        self.reward_distance_scale = self.config.reward_distance_scale

        self.control_penalty_scale=self.config.control_penalty_scale


        self.render_mode = render_mode
        self._elapsed_steps = 0
        self.current_position = None

       
        self.num_frames = self.config.num_frames
        self.obs_buffer = deque(maxlen=self.num_frames)
        self.include_actuator_lengths_in_obs = (
            self.config.include_actuator_lengths_in_obs
        )

 


        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(
                self.model, height=self.image_size[0], width=self.image_size[1]
            )
        else:
            self.renderer = None
        self.viewer = None

       
        
        self.max_distance = np.linalg.norm(
        self.target_bounds_max - self.target_bounds_min
        )

        self.prev_action=None
        self.prev_dist=None
        self.action_space = spaces.Box(
            low= -1,     
            high=1,   
            shape=(3,),
            dtype=np.float32,
        )
        self.action_dim = 3

        # Store the original control range for motor position mapping
        actuator_ctrlrange = self.model.actuator_ctrlrange
        low = actuator_ctrlrange[:, 0]
        high = actuator_ctrlrange[:, 1]
        self.actuator_low = low
        self.actuator_high = high

        # Observation space: tip_position (3) + target_position (3) + actuator_lengths
        single_frame_obs_dim = 6  # Tip pos (3) + Target pos (3)
        if self.include_actuator_lengths_in_obs:
            single_frame_obs_dim += 3

        stacked_obs_shape = (self.num_frames * single_frame_obs_dim,)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=stacked_obs_shape,
            dtype=np.float32,
        )

        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.tip_site_name
        )


        self.target_site_id = -1
        self.target_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "target"
            )
          

        self.target_position: np.ndarray = np.zeros(3)
        self.workspace_center = (
        self.target_bounds_min + self.target_bounds_max
    ) / 2.0

        self.workspace_scale = (
            self.target_bounds_max - self.target_bounds_min
        ) / 2.0
      
    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        return (pos - self.workspace_center) / self.workspace_scale


    def _normalize_actuator_lengths(self, lengths: np.ndarray) -> np.ndarray:
        return (
            2.0
            * (lengths - self.actuator_low)
            / (self.actuator_high - self.actuator_low)
            - 1.0
        )
    def _get_obs(self) -> np.ndarray:
        """Retrieves the stacked observation from the buffer."""
        assert len(self.obs_buffer) == self.num_frames, "Observation buffer not full!"
        return np.concatenate(list(self.obs_buffer), axis=0).astype(np.float32)

    def _get_current_raw_obs(self) -> np.ndarray:
        """Gets normalized observation for current state."""

        tip_position = self._get_tip_position()
        target_position = self.target_position.copy()

        # Normalize positions to roughly [-1, 1]
        tip_position = self._normalize_position(tip_position)
        target_position = self._normalize_position(target_position)

        obs_parts = [tip_position, target_position]

        if self.include_actuator_lengths_in_obs:
            actuator_lengths = self.data.actuator_length.copy().astype(np.float64)

            actuator_lengths = self._normalize_actuator_lengths(
                actuator_lengths
            )

            obs_parts.append(actuator_lengths)

        obs = np.concatenate(obs_parts).astype(np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _get_tip_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_site_id].copy()

    

    def step(self, action):
        action = np.clip(action, -1, 1)
        ctrl = self.actuator_low + (action + 1.0) * 0.5 * (self.actuator_high - self.actuator_low)
        self.data.ctrl[:] = ctrl

        
        self.data.ctrl[:] = ctrl
        # --- simulate ---
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self._is_unstable() or np.any(np.abs(self.data.qacc[:]) > 1e9):
                print(  "Unstable state detected! Ending episode.")
                return self._fail_step()

        self._elapsed_steps += 1
        mujoco.mj_forward(self.model, self.data)

        return self._compute_step()


    def _is_unstable(self):
        return (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or not np.isfinite(self.data.qacc).all()
        )


    def _fail_step(self):
        return (
            self._get_obs(),
            -10.0,
            True,
            False,
            self._get_info()
        )


    def _compute_step(self):

        tip = self._get_tip_position()
      
        dist = np.linalg.norm(
            tip - self.target_position
        )

        # Normalize distance to [0,1]
        normalized_dist = dist / self.max_distance

        normalized_dist = np.clip(
            normalized_dist,
            0.0,
            1.0
        )

        # Smooth normalized reward
        reward = np.exp(-5.0 * normalized_dist)

        truncated = (
            self._elapsed_steps >= self._max_episode_steps
        )

        obs = self._get_current_raw_obs()
        self.obs_buffer.append(obs)

        return (
            self._get_obs(),
            float(reward),
            False,
            truncated,
            self._get_info(),
        )
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.prev_action=None
        self.prev_dist=None
        self._elapsed_steps = 0
        self.obs_buffer.clear()
        

        min_val, max_val = self.initial_actuator_config

        action = self.np_random.uniform(min_val, max_val, size=3)
        action = np.clip(action, self.actuator_low, self.actuator_high)

        self.current_position = action
        self.data.ctrl[:] = action
        self.target_position = self.np_random.uniform(
            self.target_bounds_min,
            self.target_bounds_max,
            size=3
        ).astype(np.float32)

        self.data.site_xpos[self.target_site_id] = self.target_position

        mujoco.mj_forward(self.model, self.data)

        raw = self._get_current_raw_obs()
        for _ in range(self.num_frames):
            self.obs_buffer.append(raw)

        tip = self._get_tip_position()
        self.prev_dist = np.linalg.norm(tip - self.target_position)

        return self._get_obs(), self._get_info()
    def _get_info(self) -> Dict[str, Any]:
        return {
            "elapsed_steps": self._elapsed_steps,
        }

    def render(self) -> Optional[Union[np.ndarray, None]]:
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                raise RuntimeError(
                    "Renderer not initialized for rgb_array render mode."
                )
            self.renderer.update_scene(self.data, camera=self.camera_names[0])
            return self.renderer.render()
        elif self.render_mode == "human":
            self.data.site_xpos[self.target_site_id] = self.target_position.copy()
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    
def env_creator(env_config: Dict[str, Any]) -> TentacleTargetFollowingEnv:
    """Creator function for RLlib registration."""
    config = RLEnvironmentConfig(**env_config)
    render_mode = env_config.get("render_mode", None)
    return TentacleTargetFollowingEnv(config=config, render_mode=render_mode)