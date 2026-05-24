#Taken from: https://github.com/CodeReclaimers/neat-python/blob/master/examples/single-pole-balancing/evolve-feedforward.py

"""
Single-pole balancing experiment using a feed-forward neural network.
"""

import multiprocessing
import os
import pickle
import neat
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

runs_per_net = 2 #Was originally 5

#We will basically never modify run() and eval_genomes(). The only thing we will modify is eval_genome() - Running the environment, getting rewards, modfying algorithm.

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make("BipedalWalker-v3")
        observation, info = env.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        done = False
        while not done:

            action = net.activate(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated #If the pole is too far or the cart is too far, we are done.
            fitness += reward

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # Setup the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    fitness_history = []
    line, = ax.plot([], [])
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('NEAT Training Progress')

    # Create a custom reporter to update the plot
    class PlotReporter(neat.reporting.BaseReporter):
        def post_evaluate(self, config, population, species, best_genome):
            fitness_history.append(best_genome.fitness)
            line.set_xdata(range(len(fitness_history)))
            line.set_ydata(fitness_history)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)

    pop.add_reporter(PlotReporter())

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Final plot update and keep the window open
    plt.ioff()
    plt.show()

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    #Comment this out, this basically just visualizes the NN.

    # visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    # visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    # node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    # visualize.draw_net(config, winner, True, node_names=node_names)

    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-feedforward.gv")
    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)


if __name__ == '__main__':
    run()