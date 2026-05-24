from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from environment import make_env
from config import MODELS_DIR, LOGS_DIR, HYPERPARAMETERS
import os

def setup_callbacks(env):
    """Setup training callbacks"""
    eval_callback = EvalCallback(
        env,
        best_model_save_path=MODELS_DIR,
        log_path=LOGS_DIR,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    return eval_callback

def train():
    env = make_env()
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOGS_DIR
    )

    callback = setup_callbacks(env)
    
    model.learn(
        total_timesteps=HYPERPARAMETERS["total_timesteps"],
        callback=callback
    )

    model.save(os.path.join(MODELS_DIR, "final_model"))
    return model

if __name__ == "__main__":
    train()