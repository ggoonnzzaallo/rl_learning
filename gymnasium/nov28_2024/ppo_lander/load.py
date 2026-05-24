import gymnasium as gym
from stable_baselines3 import A2C, PPO

env = gym.make("LunarLander-v3", render_mode="human")
env.reset()

models_dir = "models/PPO"
model_path = f"{models_dir}/25000.zip"

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()

#For viz, run this in terminal:
#tensorboard --logdir=logs


#ep_len_mean = episode length mean
#ep_rew_mean = episode reward mean