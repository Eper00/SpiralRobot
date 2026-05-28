from matplotlib import pyplot as plt

from common.base_class import TentacleBaseEnv
from common.support import _get_sites_positions, _normalize_actuator_lengths,_normalize_position
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
from gymnasium import spaces
from imitation.data.types import Trajectory
from imitation.algorithms.bc import BC
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from pathlib import Path
from datetime import datetime
from imitation.util import logger as imit_logger
import torch
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
    def workspace_edge_polcy(self,direction,coiling_bias):
       
        if direction == 1:
            self.action[0] = -0.9
            self.action[1] = coiling_bias
        else:
            self.action[0] = coiling_bias
            self.action[1] = -0.9
        return np.clip(self.action, -1, 1)
    def workspace_center_polcy(self,direction,base,small_gain):
        if direction == 1:
            self.action[0] = base
            self.action[1] = base+small_gain
        else:
            self.action[0] = base+small_gain
            self.action[1] = base
        return np.clip(self.action, -0.2, 0.2)

    def one_rollout(self):
        self._base_reset()

        direction = np.random.choice([0,1])
        policy_switch = np.random.choice([0,1])

        coiling_bias = np.random.uniform(-1, -0.5)
        base = np.random.uniform(-0.7,0.7)
        small_gain = np.random.uniform(-0.1,0.1)

        marker_positions = []
        actions = []
        actuators = []
        policy_switch=1
        for t in range(self._max_episode_steps):

            if policy_switch == 0:
                self.workspace_edge_polcy(direction, coiling_bias)
                policy_name = "edge"
            else:
                self.workspace_center_polcy(direction, base, small_gain)
                policy_name = "center"

            marker_positions.append(
                _get_sites_positions(
                    self.model,
                    self.data,
                    self.marker_names
                )[:, 1:]
            )

            if self.include_actuator_lengths_in_obs:
                actuator = self.data.actuator_length.copy()
                actuators.append(actuator.copy())

            actions.append(self.action.copy())

            self._base_step(self.action)

        final_tip = _get_sites_positions(
            self.model,
            self.data,
            self.marker_names[-1]
        )[0][1:]

        target = final_tip.copy()
       
        
        return  marker_positions,actuators,actions,target,policy_name
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
    def generate_demonstrations(self, amount_of_demonstration):

       
        trajectories = []

        for _ in range(amount_of_demonstration):

            marker_positions,actuators,actions,target,_=self.one_rollout()
            actions=actions[:-1]
            observations = self.assebmly_observation(
                    marker_positions,
                    actuators,
                    target
                )

            traj = Trajectory(
                obs=np.array(observations, dtype=np.float32),
                acts=np.array(actions, dtype=np.float32),
                infos=None,
                terminal=True,
            )

            trajectories.append(traj)

        return trajectories
    def train_BC(self, amount_of_demonstration):

        trajectories = self.generate_demonstrations(amount_of_demonstration)

        policy = ActorCriticPolicy(
        observation_space=self.observation_space,
        action_space=self.action_space,
        lr_schedule=lambda _: self.lr,
        net_arch=self.net_arch,
        activation_fn=self.activation_fn
        )
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_bc")
        root = Path("results") / run_name
        model_dir = root / "models"
        log_dir = root / "logs"

        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = imit_logger.configure(
        folder=str(log_dir),
        format_strs=["tensorboard", "stdout"],
    )
        bc_trainer = BC(
            observation_space=self.observation_space,
            action_space=self.action_space,
            rng=self.seed,
            demonstrations=trajectories,
            policy=policy,
            custom_logger=logger
        )

        bc_trainer.train(n_epochs=self.bc_epochs)
        torch.save(
        bc_trainer.policy.state_dict(),
        model_dir / "bc_policy.pt"
        )

        print(f"Done. Results saved to: {root}")
        return bc_trainer
        
    def many_rollout(self, amount_of_demonstration):

        results = [
            self.one_rollout()
            for _ in range(amount_of_demonstration)
        ]

        marker_positions = [r[0] for r in results]
        actuators = [r[1] for r in results]
        actions = [r[2] for r in results]
        targets = [r[3] for r in results]
        policy_names = [r[4] for r in results]

        # target pontok (XY)
        points = np.array([t for t in targets])

        labels = np.array(policy_names)

        for label in set(labels):
            mask = labels == label

            plt.scatter(
                points[mask, 0],
                points[mask, 1],
                label=label
            )

        plt.legend()
        plt.show()