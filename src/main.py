import numpy as np
import matplotlib.pyplot as plt

from geneticalgorithm import *
from game import *

def moving_average(a, n=3):
    kernel = np.ones(n) / n
    return np.convolve(a, kernel, mode='same')


LAYERS = [
    [4 + 2 + 4, 16, ELU],
    [None, 16, ELU],
    [None, 16, ELU],
    [None, 4, identity]
]

if __name__ == '__main__':
    ga = GeneticAlgorithm(
        layers=LAYERS,
        population_size=2000,
        num_generations=150,
        num_repeats_per_agent=1,
        game_fps=0, # Uncapped
        snake_moves_per_second=7
    )

    ga.execute(p_selection=0.05, p_mutation = 0.05, std_mutation=0.1)

    data = np.array(ga.generation_stats)
    generations = data[:, 0]
    best_fitness = data[:, 1]
    best_fitness = best_fitness/np.max(best_fitness)

    mean_ticks_survived = data[:, 2]
    mean_ticks_survived = mean_ticks_survived/np.max(mean_ticks_survived)

    best_fitness_avg_of_n = moving_average(best_fitness, n=5)
    mean_ticks_survived_avg_of_n = moving_average(mean_ticks_survived, n=5)

    plt.title('Improvements across generations')
    plt.scatter(generations, best_fitness, marker='x', label='Fitness of best agent', color='blue')
    plt.plot(generations, best_fitness_avg_of_n, linestyle='solid', marker='none', label='Moving avg', color='blue')
    plt.scatter(generations, mean_ticks_survived/np.max(mean_ticks_survived), marker='x', label='Mean number of ticks survived by best agent', color='orange')
    plt.plot(generations, mean_ticks_survived_avg_of_n, linestyle='solid', marker='none', label='Moving avg', color='orange')
    plt.xlabel('Generation')
    plt.ylabel('Normalized values')
    plt.legend()
    plt.tight_layout()
    plt.savefig('generation_stats.png')

    game = Game(
        game_fps=60,
        snake_moves_per_second=7,
        agent=None
    )

    game.loadGridRecordsFromJSON('replays/best-agent-replay.json')
    game.replay(snake_moves_per_second=7, title='Best snake replay')