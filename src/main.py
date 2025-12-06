import numpy as np
import matplotlib.pyplot as plt

from geneticalgorithm import *
from game import *
LAYERS = [
    [2 + 2 + 1 + 4, 48, ELU],
    [None, 48, ELU],
    [None, 24, ELU],
    [None, 4, identity]
]

POPULATION_SIZE = 1000

NUM_GENERATIONS = 300

if __name__ == '__main__':
    # ga = GeneticAlgorithm(
    #     layers=LAYERS,
    #     population_size=POPULATION_SIZE,
    #     num_generations=NUM_GENERATIONS,
    #     game_fps=0, # Uncapped
    #     snake_moves_per_second=7
    # )

    # ga.execute(p_selection=0.1, p_mutation = 0.2, std_mutation=0.5)

    # data = np.array(ga.generation_stats)
    # plt.title('Highest average fitness for each generation')
    # plt.plot(data[:, 0], data[:, 1], linestyle='--', marker='o')
    # plt.xlabel('Generation')
    # plt.ylabel('Highest score')
    # plt.tight_layout()
    # plt.savefig('generation_stats.png')

    game = Game(
        game_fps=GAME_FPS,
        snake_moves_per_second=SNAKE_MOVES_PER_SECOND,
        agent=None
    )

    game.loadGridRecordsFromJSON('replays/best-agent.json')
    game.replay(snake_moves_per_second=SNAKE_MOVES_PER_SECOND, title='Best snake replay')