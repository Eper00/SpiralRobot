import time

import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import mujoco.viewer
import os
from collections import deque
from typing import  Optional, Dict, Any
from common.support import _get_sites_positions, _action_to_ctrl,_normalize_position,_normalize_actuator_lengths,load_config,sample_target
import torch
class TentacleBaseEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, config, render_mode=None):

        super().__init__()
        self.config = load_config(config) if isinstance(config, str) else config
        policy_cfg=self.config['policy']
        self.bc_cfg=self.config['il']
        self.net_arch = policy_cfg["net_arch"]
        self.lr = float(policy_cfg["learning_rate"])
        activation_fn = policy_cfg["activation_fn"]
        if activation_fn == "relu":
            self.activation_fn = torch.nn.ReLU
        elif activation_fn == "tanh":
            self.activation_fn = torch.nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        self.render_delay=self.config['rl_evaluation']['render_delay']
        self.config = self.config['rl_env']
        self.render_mode = render_mode
        self.num_frames = self.config['num_frames']
        self.obs_buffer = deque(maxlen=self.num_frames)
        # -------------------------
        # XML
        # -------------------------

       
        xml_file = self.config['xml_file']
        
        if not os.path.exists(xml_file):
            script_dir = os.path.dirname(__file__)
            xml_file = os.path.join(script_dir, xml_file)

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        self.segment_effective_radius = []
        delta = 0.01   # 3 mm offset – hangolható
        self.marker_names= [f"marker_{i}" for i in range(1, self.config['marker_number']+1)]
       
        for i in range(1, len(self.marker_names)+1):   
            s1 = f"s{i}_1"
            s3 = f"s{i}_3"

            id1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s1)
            id3 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s3)

            # Y koordináták (szegmens két oldala)
            y1 = self.model.site_pos[id1][1]
            y3 = self.model.site_pos[id3][1]

            width = abs(y1 - y3)
            radius = width / 2.0

            effective_radius = radius + delta
            self.segment_effective_radius.append(effective_radius)


        
        #self.marker_names= "marker_25"
        # -------------------------
        # Config
        # -------------------------

        self.include_actuator_lengths_in_obs = (
            self.config['include_actuator_lengths_in_obs']
        )

        self.num_frames = self.config['num_frames']
        self.workspace_center = np.array(self.config['workspace_center'])  # pl. [0.0, 0.6]
        self.workspace_inner_radius = self.config['workspace_inner_radius']  # pl. 0.05
        self.workspace_outer_radius = self.config['workspace_outer_radius'] 


        self.workspace_scale = np.array([self.workspace_outer_radius,
                                    self.workspace_outer_radius])
        # -------------------------
        # Mujoco timing
        # -------------------------
        self.simulation_length_seconds = (
            self.config['simulation_length_seconds']
        )
        self.max_distance = np.linalg.norm(
        self.workspace_inner_radius
        )
        self.reward_distance_scale = (
        self.config['reward_distance_scale']
    )
        self.time_between_steps_seconds = (
            self.config['time_between_steps_seconds']
        )
 
        self.timestep = self.model.opt.timestep

        
        self.frame_skip = max(
            1,
            round(
                self.time_between_steps_seconds / self.timestep
            ),
        )
        self.time_per_step = (
            self.frame_skip * self.timestep
        )

        self._max_episode_steps = int(
            self.simulation_length_seconds
            / self.time_per_step
        )

        # -------------------------
        # Spaces
        # -------------------------
        self.actuator_dim = self.model.nu 
        self.target_dim= len(self.workspace_center)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.actuator_dim,),
            dtype=np.float32,
        )
        self.marker_positions=_get_sites_positions(self.model, self.data, self.marker_names)[:, 1:]
        self.targets=None
        self.prev_marker_positions=None
        self.prev_target_position=None
        self.actuator_low = self.model.actuator_ctrlrange[:, 0]
        self.actuator_high = self.model.actuator_ctrlrange[:, 1]
        if self.include_actuator_lengths_in_obs:
            self.actuator = self.data.actuator_length.copy()

        self.cable_min= np.array(self.config['cable_lengths'])[0]
        self.cable_max= np.array(self.config['cable_lengths'])[1]

        # -------------------------
        # Sites
        # -------------------------
        self.tip_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.marker_names[-1],
        )

        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_geom_id = self.model.geom(name="target_geom").id

        # -------------------------
        # State
        # -------------------------
        self.target_position = np.zeros(self.target_dim)

        self._elapsed_steps = 0
        
        self.viewer = None
        self.renderer = None

        # -------------------------
        # Observation dims
        # -------------------------
        
        self.single_frame_obs_dim = len(self.marker_names) * self.target_dim + self.target_dim
        self.prev_action=None
        self.prev_dist=None
        if self.include_actuator_lengths_in_obs:
            self.single_frame_obs_dim += self.actuator_dim
        stacked_obs_shape = (self.num_frames * self.single_frame_obs_dim,)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=stacked_obs_shape,
            dtype=np.float32,
        )

    
    def _get_info(self):

        return {
                "elapsed_steps": self._elapsed_steps,
            }

    def _get_current_raw_obs(self):


       
        normalized_marker_positions=_normalize_position(self.marker_positions,self.workspace_center,self.workspace_scale)
        normalized_target_position=_normalize_position(self.target_position,self.workspace_center,self.workspace_scale)
        obs_parts = np.concatenate([normalized_marker_positions.flatten(), normalized_target_position.flatten()])
        if self.include_actuator_lengths_in_obs:

            
            
            normalized_actuator=_normalize_actuator_lengths(self.actuator,self.actuator_low,self.actuator_high)
            obs_parts = np.concatenate([obs_parts, normalized_actuator.flatten()])
   


        return obs_parts
    def _base_reset(self):

        mujoco.mj_resetData(self.model, self.data)

        self._elapsed_steps = 0
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0.
        self.prev_target_position=self.target_position
        random_mass = np.random.uniform(0.5,1)
        random_radius = np.random.uniform(0.01, 0.05)
        while True:
            [y,z]=sample_target(self.workspace_center,self.workspace_inner_radius,self.workspace_outer_radius)
            if np.abs(y)>0.05+random_radius:
                break 
        self.target_position = np.array([y, z])



        # 2) random radius
       
      
        self.model.geom_size[self.target_geom_id][0] = random_radius
        self.model.body_mass[self.target_site_id] = random_mass
        # 3) target freejoint qpos index
        jnt_id = self.model.joint(name="target_freejoint").id
        qpos_adr = self.model.jnt_qposadr[jnt_id]

        # 4) pozíció beállítása
        self.data.qpos[qpos_adr : qpos_adr+3] = np.array([
            0.0,
            self.target_position[0],
            self.target_position[1]
        ])

        # 5) orientáció (unit quaternion)
        self.data.qpos[qpos_adr+3 : qpos_adr+7] = np.array([1, 0, 0, 0])

        mujoco.mj_forward(self.model, self.data)

    def _base_step(self, action):

        action = np.clip(action, -1, 1)

        ctrl = _action_to_ctrl(
            action,
            self.actuator_low,
            self.actuator_high,
        )

        self.data.ctrl[:] = ctrl
        if self.marker_positions is not None:
            self.prev_marker_positions=self.marker_positions
        for _ in range(self.frame_skip):

            mujoco.mj_step(self.model, self.data)

            if (self.is_unstable() or np.any(np.abs(self.data.qacc) > 1e8)):
                return False

        self._elapsed_steps += 1
        if self.include_actuator_lengths_in_obs:
            self.actuator = self.data.actuator_length.copy()
        self.marker_positions = _get_sites_positions(self.model, self.data, self.marker_names)[:, 1:]
        jnt_id = self.model.joint(name="target_freejoint").id
        qpos_adr = self.model.jnt_qposadr[jnt_id]

        self.target_position=self.data.qpos[qpos_adr : qpos_adr+3][1:]
        mujoco.mj_forward(self.model, self.data)

        return True

    def render(self) -> Optional[Union[np.ndarray, None]]:
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                raise RuntimeError(
                    "Renderer not initialized for rgb_array render mode."
                )
            self.renderer.update_scene(self.data, camera=self.camera_names[0])
            return self.renderer.render()
        elif self.render_mode == "human":
            self.data.site_xpos[self.target_site_id] = [0.0, self.target_position[0], self.target_position[1]]
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()
            if self.render_delay is not None:
                time.sleep(self.render_delay)

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    def is_unstable(self):
    
        return (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or not np.isfinite(self.data.qacc).all()
        )


    def fail_step(self):
            return (
                self._get_current_raw_obs(),
                -1000.0,
                True,
                False,
                self._get_info()
            )
def env_creator(env_config: Dict[str, Any]) -> TentacleBaseEnv:
    """Creator function for RLlib registration."""
    config = load_config(env_config.get("config_path")) if "config_path" in env_config else env_config
    render_mode = env_config.get("render_mode", None)
    return TentacleBaseEnv(config=config, render_mode=render_mode)