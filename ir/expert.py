import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import mujoco

from common.loaders import RLTrainingConfig
from common.support import _get_tip_position,_action_to_ctrl,_read_dataset,_get_targets_tips


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from typing import Tuple, Dict, Any, Optional, Union
from imitation.data.types import Trajectory
class TentacleTargetFollowingClone(gym.Env):
    def __init__(
        self,
        config: Optional[RLTrainingConfig] = None,
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
        self.demonstration_number = self.config.demonstration_number
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
        self.pseudo_random_time=self.config.pseudo_random_time
        self.future_horizon = self.config.future_horizon
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0.19

        mujoco.mj_forward(self.model, self.data)

        # ---------------------------------
        # warm-up random dynamics
        # ---------------------------------
        random_action = np.random.uniform(-1, 1, 3)

        random_control = _action_to_ctrl(
            random_action,
            self.actuator_low,
            self.actuator_high
        )

        self.data.ctrl[:] = random_control

        for _ in range(self.pseudo_random_time):
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        # ---------------------------------
        # initial tip
        # ---------------------------------
        self.tip_initial = _get_tip_position(self.model, self.data)

        # ---------------------------------
        # rollout to define local horizon target
        # ---------------------------------
        tip = self.tip_initial.copy()

        for _ in range(self.future_horizon):

            action = np.random.uniform(-1, 1, 3)

            ctrl = _action_to_ctrl(
                action,
                self.actuator_low,
                self.actuator_high
            )

            self.data.ctrl[:] = ctrl

            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)

            mujoco.mj_forward(self.model, self.data)

            tip = _get_tip_position(self.model, self.data)

        # ---------------------------------
        # LOCAL GOAL (future state)
        # ---------------------------------
        self.target_position = tip.copy()

        return self._get_obs()
    def _get_obs(self) -> np.ndarray:

        tip = _get_tip_position(self.model, self.data).astype(np.float32)

        target = self.target_position.astype(np.float32)

        actuator = self.data.actuator_length.copy().astype(np.float32)

        return np.concatenate([
            tip,
            target,
            actuator
        ]).astype(np.float32)

    def _get_trajectory(self):

        mujoco.mj_resetData(self.model, self.data)

        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0.19

        tips = []
        actions = []
        actuator_lengths_list = []

        # -----------------------------
        # rollout
        # -----------------------------
        for _ in range(self._max_episode_steps):

            tip = _get_tip_position(
                self.model,
                self.data
            )

            actuator_lengths_list.append(
                self.data.actuator_length.copy()
            )

            action = np.random.uniform(-1, 1, 3)

            actions.append(action.copy())
            tips.append(tip.copy())

            ctrl = _action_to_ctrl(
                action,
                self.actuator_low,
                self.actuator_high
            )

            for _ in range(self.frame_skip):

                self.data.ctrl[:] = ctrl
                mujoco.mj_step(self.model, self.data)

        tips = np.array(tips, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)

        states = []

       

        for i in range(len(tips)):

            future_idx = min(
                i + self.future_horizon,
                len(tips) - 1
            )

            target = tips[future_idx]

            obs_parts = [
                tips[i],
                target,
            ]

            if self.include_actuator_lengths_in_obs:

                obs_parts.append(
                    actuator_lengths_list[i]
                )

            states.append(
                np.concatenate(obs_parts).astype(np.float32)
            )

        states = np.array(states, dtype=np.float32)

        # T+1 requirement
        states = np.vstack([
            states,
            states[-1]
        ])

        return states, actions

    def generate_demonstrations(self):

        trajectories = []

        all_states = []
        all_actions = []

        for traj_idx in range(self.demonstration_number):

            # -----------------------------------------
            # generate single trajectory
            # -----------------------------------------
            states, actions = self._get_trajectory()

            # -----------------------------------------
            # BC trajectory object
            # IMPORTANT:
            # len(obs) == len(actions) + 1
            # -----------------------------------------
            traj = Trajectory(
                obs=states,
                acts=actions,
                infos=np.array(
                    [{} for _ in range(len(actions))],
                    dtype=object
                ),
                terminal=True,
            )

            trajectories.append(traj)

            # -----------------------------------------
            # save for visualization
            # -----------------------------------------
            all_states.append(states)
            all_actions.append(actions)

        # ---------------------------------------------
        # store all demonstrations
        # ---------------------------------------------
        self.demonstration_states = np.array(
            all_states,
            dtype=np.float32
        )

        self.demonstration_actions = np.array(
            all_actions,
            dtype=np.float32
        )

        print(
            "Generated demonstrations:"
        )

        print(
            "States:",
            self.demonstration_states.shape
        )

        print(
            "Actions:",
            self.demonstration_actions.shape
        )

        return trajectories

    def visualize_actions(self):

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        for states, actions in zip(
            self.demonstration_states,
            self.demonstration_actions
        ):

            step = max(1, len(actions) // 50)

            idx = np.arange(0, len(actions), step)

            ax.quiver(
                states[idx, 0],
                states[idx, 1],
                states[idx, 2],
                actions[idx, 0],
                actions[idx, 1],
                actions[idx, 2],
                length=0.03,
                normalize=True,
                alpha=0.5
            )
     
        ax.set_title("All Demonstration Actions")

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
    config = RLTrainingConfig(**env_config)
    render_mode = env_config.get("render_mode", None)
    return TentacleTargetFollowingClone(config=config, render_mode=render_mode)