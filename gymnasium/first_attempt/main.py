import gymnasium as gym
import pygame
import numpy as np
import neat
import os
import time
import matplotlib.pyplot as plt

def eval_genome(genome, config, genome_id):
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = gym.make("BipedalWalker-v3", render_mode=None)
        observation, _ = env.reset()
        
        total_reward = 0
        steps = 0
        done = False
        start_time = time.time()
        early_progress_threshold = 5  # 5 seconds to get some reward
        
        while not done and steps < 1000:
            current_time = time.time() - start_time
            
            # Early termination check
            if current_time > early_progress_threshold and total_reward <= 0:
                total_reward -= 50  # Penalty for no early progress
                break
                
            # Regular timeout check
            if current_time > 60:
                break
                
            action = net.activate(observation)
            action = np.clip(action, -1, 1)
            
            try:
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            except Exception as e:
                print(f"Error during step: {e}")
                break
        
        env.close()
        if steps >= 1000 or time.time() - start_time > 60:
            total_reward -= 100
            
        return genome_id, total_reward
        
    except Exception as e:
        print(f"Error evaluating genome {genome_id}: {e}")
        return genome_id, -1000

def eval_genomes(genomes, config):
    results = []
    for genome_id, genome in genomes:
        fitness = eval_genome(genome, config, genome_id)
        genome.fitness = fitness[1]  # fitness is returned as (genome_id, reward)

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    pop = neat.Population(config)
    
    # Add reporters to show progress
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    try:
        best_fitness = -float('inf')
        for generation in range(50):  # 50 generations
            # Run one generation
            genomes = list(pop.population.items())
            eval_genomes(genomes, config)
            
            # Find best performer
            current_best = max(g.fitness for _, g in genomes)
            best_genome = max(genomes, key=lambda x: x[1].fitness)[1]
            
            # If we found a better performer, show it
            if current_best > best_fitness:
                best_fitness = current_best
                print(f"\nNew best fitness in generation {generation}: {best_fitness}")
                print("Visualizing best performer...")
                visualize_genome(best_genome, config)
            
            # Let the population advance
            pop.run(eval_genomes, 1)
        
        # Return the winner
        return best_genome
        
    except KeyboardInterrupt:
        print("\nUser interrupted evolution")
        return None
    except Exception as e:
        print(f"\nError during evolution: {e}")
        return None

def visualize_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make("BipedalWalker-v3", render_mode="human")
    observation, _ = env.reset()
    
    total_reward = 0
    done = False
    
    while not done:
        action = net.activate(observation)
        action = np.clip(action, -1, 1)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    env.close()
    return total_reward

def plot_stats(stats):
    # Plot average and best fitness over generations
    generation = range(len(stats.most_fit_genomes))
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = np.array(stats.get_fitness_mean())
    
    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'b-', label='Best Fitness')
    plt.plot(generation, avg_fitness, 'r-', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Get the path to the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-bipedal")
    
    winner = run_neat(config_path)
    if winner:
        print("\nBest genome:\n{!s}".format(winner))
        
        # Load the config to create the network
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        
        # Plot the stats
        stats = neat.StatisticsReporter()
        plot_stats(stats)
        
        # Visualize the best performer
        print("\nVisualizing best performer...")
        final_score = visualize_genome(winner, config)
        print(f"Final score: {final_score}")