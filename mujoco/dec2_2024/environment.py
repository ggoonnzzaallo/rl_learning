import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(render_mode=None):
    """Create and wrap the Ant environment"""
    env = gym.make('Ant-v5', render_mode=render_mode)
    env = DummyVecEnv([lambda: env])
    return env