import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
import time 

# Create directories for logging and saving models
models_dir = f"models/PPO-{int(time.time())}"
logs_dir = f"logs/PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Create timestamp for unique run identification
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"PPO_ant_{timestamp}"

# Create the environment with Monitor wrapper for logging
env = gym.make('Ant-v5', render_mode=None)
env = Monitor(env, os.path.join(logs_dir, run_name))
env = DummyVecEnv([lambda: env])
#Read here for Ant-v5 docs: https://gymnasium.farama.org/environments/mujoco/ant/

# Create the PPO model with tensorboard logging
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log=logs_dir  # Enable tensorboard logging
)

# Create an evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path=os.path.join(models_dir, run_name),
    log_path=os.path.join(logs_dir, run_name),
    eval_freq=1000,
    deterministic=True,
    render=False
)

# Create a checkpoint callback to save models periodically
checkpoint_callback = CheckpointCallback(
    save_freq=1000,  # Save every 1000 steps (same as eval_freq)
    save_path=os.path.join(models_dir, run_name),
    name_prefix="model",
    save_replay_buffer=True,
    save_vecnormalize=True
)

# Combine callbacks
callbacks = CallbackList([checkpoint_callback, eval_callback])

# Train the model with combined callbacks
model.learn(
    total_timesteps=1000000,
    callback=callbacks,  # Use the callback list instead of single callback
    tb_log_name=run_name
)

# Save the final model
model.save(os.path.join(models_dir, f"{run_name}_final"))

# Test the trained model
eval_env = gym.make('Ant-v5', render_mode="human")
eval_env = DummyVecEnv([lambda: eval_env])
obs, _ = eval_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, _ = eval_env.reset()

eval_env.close()