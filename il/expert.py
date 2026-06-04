from matplotlib import pyplot as plt

from common.base_class import TentacleBaseEnv
from common.support import _get_sites_positions, _normalize_actuator_lengths,_normalize_position
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
from gymnasium import spaces
from imitation.data.types import Trajectory

class TentacleTargetFollowingExpert(TentacleBaseEnv):
    def __init__(
        self,
        config: Dict[str, Any]=None,
        render_mode: str = None,
    ):
        super().__init__(config, render_mode)
        self.action = np.zeros(self.actuator_dim)
        self.seed=self.config['seed']
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.single_frame_obs_dim,),
            dtype=np.float32,
        )
        self.bc_epochs=self.bc_cfg['BC_epochs']
        self.bc_batch_size=self.bc_cfg['BC_batch_size']
        self.trajectories=self.bc_cfg['trajectories']
        self.sample_for_one_trajectory=self.bc_cfg['sample_for_one_trajectory']

    def coiling_policy(self,direction):
        if direction == 0:
            self.action=[-1,1]
        else:
            self.action=[1,-1]
    def random_policy(self):
        self.action=np.random.uniform(-1,1,size=self.actuator_dim)
    def one_trajectory(self):
        self._base_reset()

        direction = np.random.choice([0,1])
       

    
        tips=[]
        actions = []
        actuators = []
        for _ in range(50):
            tip = _get_sites_positions(
                    self.model,
                    self.data,
                    self.marker_names[-1]
                )[0][1:]

            tips.append(tip)
            if self.include_actuator_lengths_in_obs:
                actuator = self.data.actuator_length.copy()
                actuators.append(actuator.copy())

            actions.append(self.action.copy())
            self.coiling_policy(direction)
            self._base_step(self.action)
        self.random_policy()
        for t in range(self._max_episode_steps):
            tip = _get_sites_positions(
                    self.model,
                    self.data,
                    self.marker_names[-1]
                )[0][1:]

            tips.append(tip)

            if self.include_actuator_lengths_in_obs:
                actuator = self.data.actuator_length.copy()
                actuators.append(actuator.copy())

            actions.append(self.action.copy())
            self._base_step(self.action)
         


    

       
        
        return  tips,actuators,actions
    def assebmly_observation(
        self,
        marker_positions,
        actuators,
        target
    ):
      

        observations = []

        # target normalizálása egyszer
        norm_target = _normalize_position(
            target,
            self.workspace_center,
            self.workspace_scale,
        )

        for markers, actuator in zip(
            marker_positions,
            actuators
        ):

            # marker pozíciók normalizálása
            norm_markers = _normalize_position(
                markers,
                self.workspace_center,
                self.workspace_scale,
            )

            # marker + target összefűzés
            obs_parts = np.concatenate([
                norm_markers.flatten(),
                norm_target.flatten()
            ])

            # actuator hozzáadás ha kell
            if self.include_actuator_lengths_in_obs:

                actuator = np.array(actuator).flatten()

                actuator = _normalize_actuator_lengths(
                    actuator,
                    self.cable_min,
                    self.cable_max,
                )

                obs_parts = np.concatenate([
                    obs_parts,
                    actuator
                ])

    

            observations.append(obs_parts)

        return np.array(observations)

    def _make_transition(self, state_t, actuator_t, target):
        """Egyetlen HER-szerű BC minta előállítása."""
        return self.assebmly_observation(
            [state_t],
            [actuator_t],
            target
        )[0]


    def _final_observation(self, last_state, last_actuator):
        """Extra utolsó observation, hogy len(obs) = len(acts) + 1 legyen."""
        return self.assebmly_observation(
            [last_state],
            [last_actuator],
            last_state  # target itt mindegy
        )[0]


    def generate_demonstrations(self):

        trajectories = []
        targets = []

        S = self.sample_for_one_trajectory      # pl. 1000
        K = 100                                   # ennyi target egy trajból

        x_min, y_min = self.target_bounds_min
        x_max, y_max = self.target_bounds_max

        grid_n = 200
        visited_bins = np.zeros((grid_n, grid_n), dtype=bool)

        def get_bin(x, y):
            bx = int((x - x_min) / (x_max - x_min) * (grid_n - 1))
            by = int((y - y_min) / (y_max - y_min) * (grid_n - 1))
            return bx, by

        for _ in range(self.trajectories):

            # --- 1) Trajektória generálás ---
            tips, actuators, actions = self.one_trajectory()

            # --- 2) Több target kiválasztása ---
            possible_targets = list(range(40, len(tips)))
            if len(possible_targets) < K:
                continue

            target_indices = np.random.choice(possible_targets, size=K, replace=False)

            observations = []
            actions_trajectory = []

            used_time_indices = set()  # hogy ne duplikáljunk state–action párokat

            for target_idx in target_indices:

                target = tips[target_idx]

                # bin check
                bx, by = get_bin(target[0], target[1])
                if visited_bins[bx, by]:
                    continue
                visited_bins[bx, by] = True
                targets.append(target)

                # --- 3) Mintavételezés ehhez a targethez ---
                valid_indices = [i for i in range(len(tips)-1) if i not in used_time_indices]

                if len(valid_indices) == 0:
                    break

                if len(valid_indices) >= S // K:
                    sampled_indices = np.random.choice(valid_indices, size=S // K, replace=False)
                else:
                    sampled_indices = valid_indices

                used_time_indices.update(sampled_indices)

                # --- 4) Minden mintához ugyanaz a target ---
                for t in sampled_indices:
                    state_t = tips[t]
                    action_t = actions[t]
                    actuator_t = actuators[t]

                    obs_t = self._make_transition(state_t, actuator_t, target)

                    observations.append(obs_t)
                    actions_trajectory.append(action_t)

            if len(actions_trajectory) == 0:
                continue

            # extra utolsó observation
            last_obs = self._final_observation(tips[-1], actuators[-1])
            observations.append(last_obs)

            traj = Trajectory(
                obs=np.array(observations, dtype=np.float32),
                acts=np.array(actions_trajectory, dtype=np.float32),
                infos=None,
                terminal=True,
                )

            trajectories.append(traj)

        # --- Vizualizáció ---
        plt.scatter([t[0] for t in targets], [t[1] for t in targets], s=5)
        plt.title("BC Targets (Multiple targets per trajectory, no duplication)")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid()
        plt.show()

        return trajectories



   
        
  