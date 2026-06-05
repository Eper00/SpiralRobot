from matplotlib import pyplot as plt

from common.base_class import TentacleBaseEnv
from common.support import _get_sites_positions, _normalize_position,_normalize_actuator_lengths
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
        self.action = [-1, 1] if direction == -1 else [1, -1]

    def random_policy(self):
        self.action = np.random.uniform(-1, 1, size=self.actuator_dim)

    # ---------------------------------------------------------
    # TRAJECTORY GENERATION
    # ---------------------------------------------------------
    def one_trajectory(self):
        #Itt nicnsen semmilyen target
        self._base_reset()

        direction = np.random.choice([-1, 1])

        tips = []
        actions = []
        actuators = []

        # --- COILING --- nem kell target, csak a tekercseléshez szükséges akciók
        for _ in range(50):
           
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

        return tips, actuators, actions, direction

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

        x_min, y_min = self.target_bounds_min
        x_max, y_max = self.target_bounds_max

        grid_n = 100
        visited_bins = np.zeros((grid_n, grid_n), dtype=bool)

        def get_bin(x, y):
            bx = int((x - x_min) / (x_max - x_min) * (grid_n - 1))
            by = int((y - y_min) / (y_max - y_min) * (grid_n - 1))
            return bx, by

        coil_radius = 0.3
        reach_height = 0.55

        for _ in range(self.trajectories):

            tips, actuators, actions, direction = self.one_trajectory()
            T = len(tips)

            for sample in range(self.sample_for_one_trajectory):

                # --- target kiválasztása ---
                target_index = np.random.randint(self.num_frames, T)
                target = tips[target_index]

                # --- coil/reach szűrés ---
                dist = np.linalg.norm(target)
                if dist >= coil_radius:
                    if target[1] > reach_height:
                        if np.sign(target[0]) == direction:
                            continue
                    else:
                        if np.sign(target[0]) == direction:
                            continue

                # --- workspace grid szűrés ---
                bx, by = get_bin(target[0], target[1])
                if visited_bins[bx, by]:
                    continue
                visited_bins[bx, by] = True
                targets.append(target)

                # --- ÚJ trajectory minden targethez ---
                observations = []
                actions_trajectory = []

                # --- prefix trajektória ---
                for t in range(target_index - 1):

                    # RAW OBS trajectoryból
                    tip_t = tips[t]
                    actuator_t = actuators[t] if self.include_actuator_lengths_in_obs else None

                    tip_norm = _normalize_position(tip_t, self.workspace_center, self.workspace_scale)
                    target_norm = _normalize_position(target, self.workspace_center, self.workspace_scale)

                    # --- BUFFER STACK ---
                    buffer = []
                    for k in range(self.num_frames):
                        idx = max(0, t - k)

                        tip_k = tips[idx]
                        actuator_k = actuators[idx] if self.include_actuator_lengths_in_obs else None

                        tip_k_norm = _normalize_position(tip_k, self.workspace_center, self.workspace_scale)
                        raw_k = np.concatenate([tip_k_norm.flatten(), target_norm.flatten()])

                        if self.include_actuator_lengths_in_obs:
                            actuator_k_norm = _normalize_actuator_lengths(actuator_k, self.cable_min, self.cable_max)
                            raw_k = np.concatenate([raw_k, actuator_k_norm.flatten()])

                        buffer.append(raw_k)

                    buffer = buffer[::-1]
                    obs_stacked = np.concatenate(buffer)

                    observations.append(obs_stacked)
                    actions_trajectory.append(actions[t])

                # --- utolsó observation ---
                t_last = target_index - 1

                buffer = []
                for k in range(self.num_frames):
                    idx = max(0, t_last - k)

                    tip_k = tips[idx]
                    actuator_k = actuators[idx] if self.include_actuator_lengths_in_obs else None

                    tip_k_norm = _normalize_position(tip_k, self.workspace_center, self.workspace_scale)
                    target_norm = _normalize_position(target, self.workspace_center, self.workspace_scale)

                    raw_k = np.concatenate([tip_k_norm.flatten(), target_norm.flatten()])
                    if self.include_actuator_lengths_in_obs:
                        actuator_k_norm = _normalize_actuator_lengths(actuator_k, self.cable_min, self.cable_max)
                        raw_k = np.concatenate([raw_k, actuator_k_norm.flatten()])

                    buffer.append(raw_k)

                buffer = buffer[::-1]
                obs_last = np.concatenate(buffer)
                observations.append(obs_last)

                # --- trajectory mentése ---
                traj = Trajectory(
                    obs=np.array(observations, dtype=np.float32),
                    acts=np.array(actions_trajectory, dtype=np.float32),
                    infos=None,
                    terminal=True,
                )
                trajectories.append(traj)

        # vizualizáció
        plt.scatter([t[0] for t in targets], [t[1] for t in targets], s=5)
        plt.show()

        return trajectories
