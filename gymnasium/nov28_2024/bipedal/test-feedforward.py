# Taken from: https://github.com/CodeReclaimers/neat-python/blob/master/examples/single-pole-balancing/test-feedforward.py

"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

import os
import pickle

import neat
import gymnasium as gym
import numpy as np

# load the winner, this part will never change.
with open('winner-feedforward', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
# This will never change.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)
# Replace custom CartPole with Gymnasium's environment
env = gym.make('BipedalWalker-v3', render_mode="human")
observation, info = env.reset()

print()
print("Initial conditions:")
print("        x = {0:.4f}".format(observation[0]))
print("    x_dot = {0:.4f}".format(observation[1]))
print("    theta = {0:.4f}".format(observation[2]))
print("theta_dot = {0:.4f}".format(observation[3]))
print()

done = False 

while not done:
    # Get action from neural network
    action = net.activate(observation)

    # Perform action in environment
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated #If the pole is too far or the cart is too far, we are done.
    env.render()
    
    if terminated or truncated:
        break
