import numpy as np
import matplotlib.pyplot as plt

from geneticalgorithm import *
from game import *

if __name__ == '__main__':
    LAYERS = [
        [2 + 2 + 1 + 4, 48, ELU],
        [None, 48, ELU],
        [None, 24, ELU],
        [None, 4, identity]
    ]

    POPULATION_SIZE = 10000

    NUM_GENERATIONS = 100

    ga = GeneticAlgorithm(
        layers=LAYERS,
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        game_fps=0, # Uncapped
        snake_moves_per_second=7
    )

    ga.execute(p_selection=0.05, p_mutation = 0.1, std_mutation=0.1)

    data = np.array(ga.generation_stats)
    plt.title('Highest average fitness for each generation')
    plt.plot(data[:, 0], data[:, 1], linestyle='--', marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Highest score')
    plt.tight_layout()
    plt.savefig('generation_stats.png')