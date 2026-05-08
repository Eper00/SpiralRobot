import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import mujoco

from common.loaders import RLEnvironmentConfig
from common.support import _get_tip_position,_action_to_ctrl


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from typing import Tuple, Dict, Any, Optional, Union
from imitation.data.types import Trajectory
class TentacleTargetFollowingClone(gym.Env):
    def __init__(
        self,
        config: Optional[RLEnvironmentConfig] = None,
        render_mode: str = None,
    ):
        self.config = config
        self.render_mode = render_mode


        # -------------------------
        # load xml
        # -------------------------
        xml_file = "../assets/simulation/tentacle.xml"
        if not os.path.exists(xml_file):
            script_dir = os.path.dirname(__file__)
            xml_file = os.path.join(script_dir, xml_file)

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # -------------------------
        # config
        # -------------------------
        self.tip_site_name = self.config.tip_site_name
        self.include_actuator_lengths_in_obs = (
            self.config.include_actuator_lengths_in_obs
        )
        self.action_space = spaces.Box(
            low= -1,     
            high=1,   
            shape=(3,),
            dtype=np.float32,
        )
        self.num_frames = self.config.num_frames

        self.action_dim = 3
        self.demonstration_number = getattr(self.config, "demostraion_number", 20)
        single_frame_obs_dim = 6
        if self.include_actuator_lengths_in_obs:
            single_frame_obs_dim += 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(single_frame_obs_dim,),
            dtype=np.float32,
        )
        # -------------------------
        # actuator range
        # -------------------------
        self.actuator_low = self.model.actuator_ctrlrange[:, 0]
        self.actuator_high = self.model.actuator_ctrlrange[:, 1]

        # -------------------------
        # storage
        # -------------------------
        self.demonstration_states = []
        self.demonstration_actions = []

        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.tip_site_name
        )
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

    def _get_trajectory(self):

        mujoco.mj_resetData(self.model, self.data)

        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0.19

        tips = []
        actions = []

        # ---------------------------------------
        # rollout
        # ---------------------------------------
        for _ in range(self._max_episode_steps):

            action = np.random.uniform(-1, 1, 3)

            ctrl = _action_to_ctrl(
                action,
                self.actuator_low,
                self.actuator_high
            )

            for _ in range(self.frame_skip):
                self.data.ctrl[:] = ctrl
                mujoco.mj_step(self.model, self.data)

            tip = _get_tip_position(self.model, self.data)

            tips.append(tip.copy())
            actions.append(action.copy())

        tips = np.array(tips, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)

        # ---------------------------------------
        # target = final position of trajectory
        # ---------------------------------------
        target = tips[-1].copy()

        # ---------------------------------------
        # actuator state (fixed per trajectory)
        # ---------------------------------------
        actuator_lengths = None
        if self.include_actuator_lengths_in_obs:
            actuator_lengths = self.data.actuator_length.copy().astype(np.float32)

        # ---------------------------------------
        # build observations (T steps)
        # ---------------------------------------
        states = []

        for i in range(len(tips)):

            obs_parts = [
                tips[i],
                target,
            ]

            if self.include_actuator_lengths_in_obs:
                obs_parts.append(actuator_lengths)

            states.append(np.concatenate(obs_parts).astype(np.float32))

        states = np.array(states, dtype=np.float32)

        # ---------------------------------------
        # 🔥 IMPORTANT: add final obs (T+1 requirement)
        # ---------------------------------------
        final_obs = states[-1].copy()
        states = np.vstack([states, final_obs])

        return states, actions
    def build_obs(self, tip, target, actuator=None):
        obs = [tip, target]

        if self.include_actuator_lengths_in_obs:
            obs.append(actuator)

        return np.concatenate(obs).astype(np.float32)
    def generate_demonstrations(self):

        trajectories = []

        for traj_idx in range(self.demonstration_number):

            mujoco.mj_resetData(self.model, self.data)
            self.data.qvel[:] = 0
            self.data.ctrl[:] = 0.19

            tips = []
            actions = []

            for _ in range(self._max_episode_steps):

                action = np.random.uniform(-1, 1, 3)

                ctrl = _action_to_ctrl(
                    action,
                    self.actuator_low,
                    self.actuator_high
                )

                for _ in range(self.frame_skip):
                    self.data.ctrl[:] = ctrl
                    mujoco.mj_step(self.model, self.data)

                tip = _get_tip_position(self.model, self.data)

                tips.append(tip.copy())
                actions.append(action.copy())

            tips = np.array(tips, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)

            target = tips[-1].copy()

            actuator_lengths = None
            if self.include_actuator_lengths_in_obs:
                actuator_lengths = self.data.actuator_length.copy().astype(np.float32)

            obs = []

            for i in range(len(tips)):

                obs_parts = [tips[i], target]

                if self.include_actuator_lengths_in_obs:
                    obs_parts.append(actuator_lengths)

                obs.append(np.concatenate(obs_parts).astype(np.float32))

            obs = np.array(obs, dtype=np.float32)

            # 🔥 CRITICAL FIX: T+1 observation
            obs = np.vstack([obs, obs[-1]])

            traj = Trajectory(
                obs=obs,
                acts=actions,
                infos=np.array([{} for _ in range(len(actions))], dtype=object),
                terminal=True,
            )

            trajectories.append(traj)

        return trajectories

    def _visualize_trajectory_and_actions(self):    
     

        states = np.array(self.demonstration_states)
        actions = np.array(self.demonstration_actions)

        fig = plt.figure(figsize=(12, 6))

        # -------------------------
        # 1) Trajectory plot (3D)
        # -------------------------
        ax1 = fig.add_subplot(121, projection='3d')

        ax1.plot(
            states[:, 0],
            states[:, 1],
            states[:, 2],
            label="Tip trajectory",
            linewidth=2
        )

        ax1.scatter(
            states[0, 0], states[0, 1], states[0, 2],
            c='green', s=50, label="start"
        )

        ax1.scatter(
            states[-1, 0], states[-1, 1], states[-1, 2],
            c='red', s=50, label="end"
        )

        ax1.set_title("3D Trajectory (Tip)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend()

        # -------------------------
        # 2) Action visualization (3D vector field)
        # -------------------------
        ax2 = fig.add_subplot(122, projection='3d')

        # subsample to avoid clutter
        step = max(1, len(actions) // 200)

        idx = np.arange(0, len(actions), step)

        ax2.quiver(
            states[idx, 0],
            states[idx, 1],
            states[idx, 2],
            actions[idx, 0],
            actions[idx, 1],
            actions[idx, 2],
            length=0.05,
            normalize=True
        )

        ax2.set_title("Action field (3D control directions)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        plt.tight_layout()
        plt.show()
    def save_demonstration_dataset(
        self,
        save_path="demonstration_dataset.npz"
    ):

        states = np.array(self.demonstration_states, dtype=np.float32)
        actions = np.array(self.demonstration_actions, dtype=np.float32)

        np.savez_compressed(
            save_path,
            states=states,
            actions=actions,
        )

        print(f"Saved demonstration dataset to: {save_path}")
        print(f"States shape: {states.shape}")
        print(f"Actions shape: {actions.shape}")



def env_creator(env_config: Dict[str, Any]) -> TentacleTargetFollowingClone:
    """Creator function for RLlib registration."""
    config = RLEnvironmentConfig(**env_config)
    render_mode = env_config.get("render_mode", None)
    return TentacleTargetFollowingClone(config=config, render_mode=render_mode)