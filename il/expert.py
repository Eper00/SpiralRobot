from matplotlib import pyplot as plt

from common.base_class import TentacleBaseEnv
from common.support import _get_sites_positions
from typing import Dict, Any
import numpy as np
from imitation.data.types import Trajectory
import mujoco

class TentacleTargetFollowingExpert(TentacleBaseEnv):
    def __init__(self, config: Dict[str, Any] = None, render_mode: str = None):
        super().__init__(config, render_mode)

        self.action = np.zeros(self.actuator_dim)
        self.seed = self.config['seed']

        self.bc_epochs = self.bc_cfg['BC_epochs']
        self.bc_batch_size = self.bc_cfg['BC_batch_size']
        self.trajectories = self.bc_cfg['trajectories']
        self.sample_for_one_trajectory = self.bc_cfg['sample_for_one_trajectory']

 

    # ---------------------------------------------------------
    # POLICIES
    # ---------------------------------------------------------
    def coiling_policy(self, direction):
        self.action = [-1, 1] if direction == 0 else [1, -1]

    def random_policy(self):
        self.action = np.random.uniform(-1, 1, size=self.actuator_dim)

    # ---------------------------------------------------------
    # TRAJECTORY GENERATION
    # ---------------------------------------------------------
    def one_trajectory(self):
        self._base_reset()

        direction = np.random.choice([0, 1])

        tips = []
        actions = []
        actuators = []

        # --- COILING ---
        for _ in range(50):
            tip = _get_sites_positions(self.model, self.data, self.marker_names[-1])[0][1:]
            tips.append(tip)

            if self.include_actuator_lengths_in_obs:
                actuators.append(self.data.actuator_length.copy())

            actions.append(self.action.copy())
            self.coiling_policy(direction)
            self._base_step(self.action)

        # --- RANDOM MOTION ---
        self.random_policy()
        for _ in range(self._max_episode_steps):
            tip = _get_sites_positions(self.model, self.data, self.marker_names[-1])[0][1:]
            tips.append(tip)

            if self.include_actuator_lengths_in_obs:
                actuators.append(self.data.actuator_length.copy())

            actions.append(self.action.copy())
            self._base_step(self.action)

        return tips, actuators, actions

    # ---------------------------------------------------------
    # STACKED OBSERVATION (ugyanaz, mint RL-ben)
    # ---------------------------------------------------------
    def _stacked_obs(self):
        return np.concatenate(list(self.obs_buffer)).astype(np.float32)

    # ---------------------------------------------------------
    # DEMONSTRATION GENERATION
    # ---------------------------------------------------------
    def generate_demonstrations(self):

        trajectories = []
        targets = []

        S = self.sample_for_one_trajectory
        K = 100

        x_min, y_min = self.target_bounds_min
        x_max, y_max = self.target_bounds_max

        grid_n = 200
        visited_bins = np.zeros((grid_n, grid_n), dtype=bool)

        def get_bin(x, y):
            bx = int((x - x_min) / (x_max - x_min) * (grid_n - 1))
            by = int((y - y_min) / (y_max - y_min) * (grid_n - 1))
            return bx, by

        for _ in range(self.trajectories):

            tips, actuators, actions = self.one_trajectory()
            T = len(tips)

            coiling_indices = list(range(0, 40))
            motion_indices = list(range(40, T - 1))

            if len(motion_indices) < K:
                continue

            target_indices = np.random.choice(motion_indices, size=K, replace=False)

            observations = []
            actions_trajectory = []
            used_time_indices = set()

            # ---------------------------------------------------------
            # COILING FÁZIS – target = next state
            # ---------------------------------------------------------
            self.obs_buffer.clear()

            for t in coiling_indices:
                if t >= T - 1:
                    continue

                state_t = tips[t]
                state_tp1 = tips[t + 1]
                actuator_t = actuators[t]
                action_t = actions[t]

                target = state_tp1

                # 🔥 target beállítása MuJoCo-ban → raw obs jó lesz
                self._set_target_for_bc(target)

                # 🔥 raw frame + buffer update
                raw = self._get_current_raw_obs()
                if len(self.obs_buffer) == 0:
                    for _ in range(self.num_frames):
                        self.obs_buffer.append(raw.copy())
                else:
                    self.obs_buffer.append(raw.copy())

                obs_t = self._stacked_obs()

                # bin check
                bx, by = get_bin(target[0], target[1])
                if visited_bins[bx, by]:
                    continue
                visited_bins[bx, by] = True
                targets.append(target)

                observations.append(obs_t)
                actions_trajectory.append(action_t)
                used_time_indices.add(t)

            # ---------------------------------------------------------
            # MOTION FÁZIS – target = future state
            # ---------------------------------------------------------
            for target_idx in target_indices:

                target = tips[target_idx]

                bx, by = get_bin(target[0], target[1])
                if visited_bins[bx, by]:
                    continue
                visited_bins[bx, by] = True
                targets.append(target)

                valid_indices = [i for i in motion_indices if i < target_idx and i not in used_time_indices]

                if len(valid_indices) == 0:
                    break

                samples_per_target = max(1, S // K)
                sampled_indices = (
                    np.random.choice(valid_indices, size=samples_per_target, replace=False)
                    if len(valid_indices) >= samples_per_target
                    else valid_indices
                )

                used_time_indices.update(sampled_indices)

                for t in sampled_indices:
                    state_t = tips[t]
                    actuator_t = actuators[t]
                    action_t = actions[t]

                    # 🔥 target beállítása MuJoCo-ban
                    self._set_target_for_bc(target)

                    raw = self._get_current_raw_obs()
                    self.obs_buffer.append(raw.copy())
                    obs_t = self._stacked_obs()

                    observations.append(obs_t)
                    actions_trajectory.append(action_t)

            if len(actions_trajectory) == 0:
                continue
            raw = self._get_current_raw_obs()
            self.obs_buffer.append(raw.copy())
            last_obs = self._stacked_obs()
            observations.append(last_obs)
            traj = Trajectory(
                obs=np.array(observations, dtype=np.float32),
                acts=np.array(actions_trajectory, dtype=np.float32),
                infos=None,
                terminal=True,
            )

            trajectories.append(traj)

        # ---------------------------------------------------------
        # Vizualizáció
        # ---------------------------------------------------------
        plt.scatter([t[0] for t in targets], [t[1] for t in targets], s=5)
        plt.title("BC Targets (Frame-buffered, RL-compatible)")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid()
        plt.show()

        return trajectories
    def _set_target_for_bc(self, target):
        self.target_position = target.copy()
        self.data.site_xpos[self.target_site_id] = [
            0.0,
            target[0],
            target[1],
        ]
        mujoco.mj_forward(self.model, self.data)