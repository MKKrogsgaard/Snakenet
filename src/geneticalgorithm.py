from typing import List, Dict, Union

import numpy as np
from tqdm import tqdm

from game import Game

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def unflatten(flattened_array, shapes):
    '''Unflattens an array given the array and the original shape.'''
    result = []
    index = 0
    for shape in shapes:
        size = np.prod(shape)
        result.append(flattened_array[index: index + size].reshape(shape))
        index += size
    return result

class Network():
    def __init__(self, layers):
        self.weights = []
        self.activation_functions = []

        for layer in layers:
            if layer[0] != None:
                input_size = layer[0]
            else:
                # Use the output size from the previous layer as the new input size
                input_size = layers[layers.index(layer) - 1][1]
            output_size = layer[1]
            activation_function = layer[2]

            self.weights.append(np.random.randn(input_size, output_size))
            self.activation_functions.append(activation_function)

    def forward(self, input_data):
        '''Forward propagation.'''
        # Loop through the layers
        for i in range(len(self.weights)):
            a = np.dot(input_data, self.weights[i])
            z = self.activation_functions[i](a)
            
            input_data = z # Pass the outputs from the previous layer to the next layer

        output = input_data
        return output

class Agent():
    def __init__(self, layers):
        self.neural_network = Network(layers=layers)
        self.fitness = 0
    
    def getFitness(self, game_fps, snake_moves_per_second):
        '''Makes the agent play a game of snake and returns the score.'''
        game = Game(agent=self, game_fps=game_fps, snake_moves_per_second=snake_moves_per_second)
        return game.startGame()

class GeneticAlgorithm():
    def __init__(self, layers: List, population_size: int, num_generations: int, game_fps: int, snake_moves_per_second: int):
        self.game_fps = game_fps
        self.snake_moves_per_second = snake_moves_per_second
        self.layers = layers
        self.population_size = population_size
        self.num_generations = num_generations

    def generateAgents(self, layers, population_size):
        return [Agent(layers=layers) for i in range(population_size)]

    def getAgentFitnessScores(self, agents: List[Agent]):
        results = []
        for i in tqdm(range(len(agents))):
            agent = agents[i]
            score = agent.getFitness(game_fps=self.game_fps, snake_moves_per_second=self.snake_moves_per_second)
            agent.fitness = score
            results.append(score)
        return results
    
    def selectAgents(self, agents: List[Agent], p: int):
        '''
        Selects the agents with the highest fitness score from the list of agents.
        
        :p: The percentile of agents to select. For example, p=0.1 means we select only the top 10 % of agents.
        '''

        if p > 1:
            print(f'[!] GeneticAlgorithm.selectAgents(): p cannot be greater than 1. Setting p=1.')
            p = 1
        
        scores = self.getAgentFitnessScores(agents=agents)

        agent_score_pairs = zip(agents, scores)

        sorted_pairs = sorted(agent_score_pairs, key=lambda pair: pair[1])
        sorted_agents = [pair[0] for pair in sorted_pairs]

        index = int(p * len(agents))
        top_agents = sorted_agents[0:index + 1]
        return top_agents
        

    def crossover(self, agents: List[Agent], layers: List, population_size: int):
        '''
        Performs crossover between selected agents to generate a new population.

        Works by selecting a random split index, and picking genes from the first parent up until the split and from the second parent after the split.

        Produces two children, the first child as a above, and the second with the parents switched.
        '''
        offspring = []
        
        # Each iteration will generate two children from two parents
        for i in tqdm(range((population_size - len(agents))// 2)):
            random_index1 = np.random.randint(0, len(agents))
            random_index2 = np.random.randint(0, len(agents))

            parent1 = agents[random_index1]
            parent2 = agents[random_index2]
            child1 = Agent(layers=layers)
            child2 = Agent(layers=layers)

            shapes = [arr.shape for arr in parent1.neural_network.weights]

            genes1 = np.concatenate([arr.flatten() for arr in parent1.neural_network.weights])
            genes2 = np.concatenate([arr.flatten() for arr in parent2.neural_network.weights])

            split_index = np.random.randint(0, len(genes1) - 1)

            child1_genes = np.array(genes1[0:split_index].tolist() + genes2[split_index:].tolist())
            child2_genes = np.array(genes2[0:split_index].tolist() + genes1[split_index:].tolist())

            child1.neural_network.weights = unflatten(child1_genes, shapes)
            child2.neural_network.weights = unflatten(child2_genes, shapes)

            offspring.append(child1)
            offspring.append(child2)

        agents.extend(offspring)
        return agents

    def mutate(self, agents: List[Agent], p: float):
        '''Mutates the agents by changing a random weight. p is the probability of a mutation occuring for a given agent.'''
        if not(0 <= p <= 1):
            print(f'[!] GeneticAlgorithm.mutate(): p must be in [0,1]. Setting p=1.')
            p = 1

        for agent in agents:
            if np.random.uniform(0, 1) <= p:
                weights = agent.neural_network.weights
                shapes = [arr.shape for arr in weights]

                flattened_weights = np.concatenate([arr.flatten() for arr in weights])
                random_index = np.random.randint(0, len(flattened_weights))
                flattened_weights[random_index] = np.random.randn()

                new_weights = unflatten(flattened_weights, shapes)
                agent.neural_network.weights = new_weights

        return agents
            
    def execute(self, p_selection, p_mutation):
        self.agents = self.generateAgents(layers=self.layers, population_size=self.population_size)
        for i in range(self.num_generations):
            print(f'[+] Generation: {i + 1}')
            print('Selecting agents...')
            selected_agents = self.selectAgents(agents=self.agents, p=p_selection)
            print('Performing crossover/mutation...')
            self.agents = self.crossover(agents=selected_agents, layers=self.layers, population_size=self.population_size)
            self.agents = self.mutate(agents=self.agents, p=p_mutation)
            print(f'Highest score of generation {i + 1}: {max([agent.fitness for agent in self.agents])}')


LAYERS = [
    [20*20, 100, sigmoid],
    [None, 50, sigmoid],
    [None, 25, sigmoid],
    [None, 4, ReLU]
]

POPULATION_SIZE = 100

NUM_GENERATIONS = 5

ga = GeneticAlgorithm(
    layers=LAYERS,
    population_size=POPULATION_SIZE,
    num_generations=NUM_GENERATIONS,
    game_fps=10000,
    snake_moves_per_second=10000
)

ga.execute(p_selection=0.2, p_mutation = 0.3)

