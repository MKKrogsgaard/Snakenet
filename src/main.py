import numpy as np
import matplotlib.pyplot as plt

from geneticalgorithm import *
from game import *

def moving_average(x, n=3):
    kernel = np.ones(n) / n
    res = np.convolve(x, kernel, mode='same')
    res[-n:] = x[-n:]
    return res


LAYERS = [
    [2 + 4, 10, identity],
    [None, 10, ELU],
    [None, 10, ELU],
    [None, 4, identity]
]

if __name__ == '__main__':
    ga = GeneticAlgorithm(
        layers=LAYERS,
        population_size=500,
        num_generations=100,
        num_repeats_per_agent=1,
        game_fps=0, # Uncapped
        snake_moves_per_second=7
    )

    ga.execute(p_selection=0.1, p_mutation = 0.01, std_mutation=0.1)

    data = np.array(ga.generation_stats)
    generations = data[:, 0]
    best_fitness = data[:, 1]
    best_fitness = best_fitness/np.max(best_fitness)

    mean_ticks_survived = data[:, 2]
    mean_ticks_survived = mean_ticks_survived/np.max(mean_ticks_survived)

    best_fitness_avg_of_n = moving_average(best_fitness, n=5)
    mean_ticks_survived_avg_of_n = moving_average(mean_ticks_survived, n=5)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig.suptitle('Improvements across generations')

    ax1.scatter(generations, best_fitness, marker='x', label='Fitness of best agent', color='blue')
    ax1.plot(generations, best_fitness_avg_of_n, linestyle='solid', marker='none', label='Moving avg', color='blue')
    ax1.set_ylabel('Normalized fitness')
    ax1.legend(loc='best')
    ax1.grid(True)

    ax2.scatter(generations, mean_ticks_survived, marker='x', label='Mean ticks survived', color='orange')
    ax2.plot(generations, mean_ticks_survived_avg_of_n, linestyle='solid', marker='none', label='Moving avg', color='orange')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Normalized mean ticks survived')
    ax2.legend(loc='best')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('generation_stats.png')

    game = Game(
        game_fps=60,
        snake_moves_per_second=7,
        agent=None
    )

    game.loadGridRecordsFromJSON('replays/most-recent-agent-replay.json')
    game.replay(snake_moves_per_second=7, title='Best snake replay')