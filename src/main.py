import numpy as np
import matplotlib.pyplot as plt

from geneticalgorithm import *
from game import *

LAYERS = [
    [4 + 2 + 4, 16, ELU],
    [None, 16, ELU],
    [None, 16, ELU],
    [None, 4, identity]
]

if __name__ == '__main__':
    ga = GeneticAlgorithm(
        layers=LAYERS,
        population_size=200,
        num_generations=20,
        num_repeats_per_agent=3,
        game_fps=0, # Uncapped
        snake_moves_per_second=7
    )

    ga.execute(p_selection=0.1, p_mutation = 0.05, std_mutation=0.3)

    data = np.array(ga.generation_stats)
    generations = data[:, 0]
    highest_avg_fitness = data[:, 1]
    mean_ticks_survived = data[:, 2]

    plt.title('Improvements across generations')
    plt.plot(generations, highest_avg_fitness/np.max(highest_avg_fitness), linestyle='--', marker='o', label='Fitness of best agent')
    plt.plot(generations, mean_ticks_survived/np.max(mean_ticks_survived), linestyle='--', marker='o', label='Mean number of ticks survived by best agent')
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