import numpy as np
from imitation.data.types import Trajectory
from common.support import _get_tip_position,_normalize_position,_normalize_actuator_lengths
import copy
class Expert:

    def __init__(self, env):
        self.env = copy.deepcopy(env)
    def policy(self):
        return np.random.uniform(-1,1,3)

    def rollout_episode(self):

        self.env._base_reset()

        tips = []
        actions = []
        actuators = []

        action = self.policy()
        done = False

        while not done:
            tips.append(_get_tip_position(self.env.model, self.env.data).copy())
            actions.append(action.copy())
            if self.env.include_actuator_lengths_in_obs:
                actuators.append(self.env.data.actuator_length.copy())
            ok = self.env._simulate(action)
            if not ok:
                break

            done = self.env._elapsed_steps >= self.env._max_episode_steps

        final_tip = _get_tip_position(self.env.model, self.env.data).copy()
        tips.append(final_tip)

        if self.env.include_actuator_lengths_in_obs:
            actuators.append(self.env.data.actuator_length.copy())

        target = final_tip.copy()
        obs_array = self.build_il_obs_sequence(tips, target, actuators)
        
        return obs_array, np.array(actions, dtype=np.float32),target.copy()
    def build_il_obs_sequence(self, tips, target, actuators=None):

        target_norm = _normalize_position(
            target,
            self.env.workspace_center,
            self.env.workspace_scale
        )

        obs_seq = []

        for i in range(len(tips)):

            tip_n = _normalize_position(
                tips[i],
                self.env.workspace_center,
                self.env.workspace_scale
            )

            obs_parts = [tip_n, target_norm]

            if actuators is not None:
                actuator_n = _normalize_actuator_lengths(
                    actuators[i],
                    self.env.actuator_low,
                    self.env.actuator_high,
                )
                obs_parts.append(actuator_n)

            obs_seq.append(np.concatenate(obs_parts).astype(np.float32))

        return np.array(obs_seq, dtype=np.float32)
    def create_demonstrations(self, num_episodes=3):

        trajectories = []

        for _ in range(num_episodes):

            obs_list, act_list,target = self.rollout_episode()
           
            if len(obs_list) < 2:
                continue

            trajectories.append(
                Trajectory(
                    obs=np.array(obs_list, dtype=np.float32),
                    acts=np.array(act_list, dtype=np.float32),
                    infos=None,
                    terminal=True,
                )
            )

        return trajectories