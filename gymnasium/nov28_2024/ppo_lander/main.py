#Based on this tutorial: https://www.youtube.com/watch?v=XbWhJdQgi7E
#Relevant docs: https://stable-baselines3.readthedocs.io/en/master/

'''
Relevant ML Terms:

Environment:
What are you trying to solve? (i.e cartpole, lunar lander, etc.)
If you're trying to make some AI play a game, the game is the environment.

Model:
What algorithm are you using (PPO, SAC, TRPO, TD3, etc.)

Agent:
The entity that interacts with the environment using an algorithm/model.
For example, the cart in cartpole is the agent.

Observation: (Also known as "state")
Important details of the environment that are fed to the model to make action predictions.

Action:
What the agent does in the environment per step. "Go left" could be an example of an action.

Step:
Progress in the environment. Can be thought of like FPS, where each "frame" is a step.
In general a "step" in the environment will take an action for the agent to do, and return a new observation and reward for the step.

An action space can be either:
-Discrete: Clear classifications, go left or go right.
OR 
-Continuous: Like regression, go 0.02 right or 0.5 or -0.344221.
Typically continuous is harder to learn.
Sometimes discrete problems are treated as continuous, i.e servos that have a ton of possible discrete positions, but are treated as a continuous signal.

'''

#For rewards and stuff, see here: https://gymnasium.farama.org/environments/box2d/lunar_lander/

import gymnasium as gym
from stable_baselines3 import A2C, PPO
import os

models_dir = "models/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v3", render_mode="human")
env.reset()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 1000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

# print("sample action:", env.action_space.sample())
# print("observation space shape", env.observation_space.shape)
# print("sample observation:", env.observation_space.sample())

episodes = 10

# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         env.render()

#         obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#         done = terminated or truncated

    # print(reward)

env.close()

