import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
env_id = "CartPole-v1"
num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
env = gym.make(env_id, render_mode="human")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25_000)

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()