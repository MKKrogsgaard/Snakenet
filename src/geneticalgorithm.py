from typing import List, Dict, Union

import matplotlib.pyplot as plt
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
        self.biases = []
        self.activation_functions = []

        for index, layer in enumerate(layers):
            if layer[0] != None:
                input_size = layer[0]
            else:
                # Use the output size from the previous layer as the new input size
                input_size = layers[index - 1][1]
            output_size = layer[1]
            activation_function = layer[2]

            self.weights.append(np.random.randn(input_size, output_size))
            self.biases.append(np.random.randn(output_size))
            self.activation_functions.append(activation_function)

    def forward(self, input_data):
        '''Forward propagation.'''
        # Loop through the layers
        for i in range(len(self.weights)):
            a = np.dot(input_data, self.weights[i]) + np.array(self.biases[i])
            z = self.activation_functions[i](a)
            
            input_data = z # Pass the outputs from the previous layer to the next layer

        output = input_data
        return output

class Agent():
    def __init__(self, layers):
        self.neural_network = Network(layers=layers)
        self.fitness = 0
        self.max_loop_iterations = 1000
        self.iteration_counter = 0
    
    def getFitness(self, game_fps, snake_moves_per_second):
        '''Makes the agent play a game of snake and returns the score.'''
        self.iteration_counter = 0

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

        sorted_pairs = sorted(agent_score_pairs, key=lambda pair: pair[1], reverse=True)
        sorted_agents = [pair[0] for pair in sorted_pairs]

        index = max(1, int(p * len(agents)))
        top_agents = sorted_agents[0:index]
        return top_agents
        

    def crossover(self, agents: List[Agent], layers: List, population_size: int):
        '''
        Performs crossover between selected agents to generate a new population.

        Works by selecting a parent by coinflip, and picking the current gene from that parent, until all genes are picked.
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

            weight_shapes = [arr.shape for arr in parent1.neural_network.weights]
            
            bias_shapes = [arr.shape for arr in parent1.neural_network.biases]

            parent1_weight_genes = np.concatenate([arr.flatten() for arr in parent1.neural_network.weights])
            parent2_weight_genes = np.concatenate([arr.flatten() for arr in parent2.neural_network.weights])

            parent1_bias_genes = np.concatenate([arr.flatten() for arr in parent1.neural_network.biases])
            parent2_bias_genes = np.concatenate([arr.flatten() for arr in parent2.neural_network.biases])

            length_of_weights = len(parent1_weight_genes)
            length_of_biases = len(parent1_bias_genes)

            child1_weight_genes = np.zeros(length_of_weights)
            child2_weight_genes = np.zeros(length_of_weights)

            child1_bias_genes = np.zeros(length_of_biases)
            child2_bias_genes = np.zeros(length_of_biases)

            weight1_coinflips = np.random.randint(0, 2, size=length_of_weights)
            weight2_coinflips = np.random.randint(0, 2, size=length_of_weights)

            for j in range(length_of_weights):
                if weight1_coinflips[j] == 0:
                    child1_weight_genes[j] = parent1_weight_genes[j]
                else:
                    child1_weight_genes[j] = parent2_weight_genes[j]

                if weight2_coinflips[j] == 0:
                    child2_weight_genes[j] = parent1_weight_genes[j]
                else:
                    child2_weight_genes[j] = parent2_weight_genes[j]

            bias1_coinflips = np.random.randint(0, 2, size=length_of_biases)
            bias2_coinflips = np.random.randint(0, 2, size=length_of_biases)

            for j in range(length_of_biases):
                if bias1_coinflips[j] == 0:
                    child1_bias_genes[j] = parent1_bias_genes[j]
                else:
                    child1_bias_genes[j] = parent2_bias_genes[j]
                
                if bias2_coinflips[j] == 0:
                    child2_bias_genes[j] = parent1_bias_genes[j]
                else:
                    child2_bias_genes[j] = parent2_bias_genes[j]

            child1.neural_network.weights = unflatten(child1_weight_genes, weight_shapes)
            child2.neural_network.weights = unflatten(child2_weight_genes, weight_shapes)

            child1.neural_network.biases = unflatten(child1_bias_genes, bias_shapes)
            child2.neural_network.biases = unflatten(child2_bias_genes, bias_shapes)

            offspring.append(child1)
            offspring.append(child2)

        agents.extend(offspring)
        return agents

    def mutate(self, agents: List[Agent], p: float):
        '''Mutates the agents by changing a random weight. p is the probability of a mutation occuring for a given weight/bias.'''
        if not(0 <= p <= 1):
            print(f'[!] GeneticAlgorithm.mutate(): p must be in [0,1]. Setting p=1.')
            p = 1

        for i in tqdm(range(len(agents))):
            agent = agents[i]

            weights = agent.neural_network.weights
            weight_shapes = [arr.shape for arr in weights]

            biases = agent.neural_network.biases
            bias_shapes = [arr.shape for arr in biases]

            flattened_weights = np.concatenate([arr.flatten() for arr in weights])

            flattened_biases = np.concatenate([arr.flatten() for arr in biases])

            for j in range(len(flattened_weights)):
                if np.random.uniform(0, 1) <= p:
                    flattened_weights[j] += np.random.randn() * 0.1

            for j in range(len(flattened_biases)):
                if np.random.uniform(0, 1) <= p:
                    flattened_biases[j] += np.random.randn() * 0.1

            new_weights = unflatten(flattened_weights, weight_shapes)
            new_biases = unflatten(flattened_biases, bias_shapes)

            agent.neural_network.weights = new_weights
            agent.neural_network.biases = new_biases

        return agents
            
    def execute(self, p_selection, p_mutation):
        self.generation_stats = []

        self.agents = self.generateAgents(layers=self.layers, population_size=self.population_size)
        for i in range(self.num_generations):
            print(f'[+] Generation: {i + 1}')
            print('Selecting agents...')
            selected_agents = self.selectAgents(agents=self.agents, p=p_selection)

            highest_score = max([agent.fitness for agent in self.agents])
            self.generation_stats.append([i + 1, highest_score])

            print('Performing crossover...')
            self.agents = self.crossover(agents=selected_agents, layers=self.layers, population_size=self.population_size)
            print('Performing mutation...')
            self.agents = self.mutate(agents=self.agents, p=p_mutation)
            print(f'[+] Highest score of generation {i + 1}: {highest_score}')

            
LAYERS = [
    [20*20, 100, ReLU],
    [None, 100, ReLU],
    [None, 50, ReLU],
    [None, 25, ReLU],
    [None, 4, ReLU]
]

POPULATION_SIZE = 2000

NUM_GENERATIONS = 10

ga = GeneticAlgorithm(
    layers=LAYERS,
    population_size=POPULATION_SIZE,
    num_generations=NUM_GENERATIONS,
    game_fps=0, # Uncapped
    snake_moves_per_second=7
)

ga.execute(p_selection=0.1, p_mutation = 0.1)

data = np.array(ga.generation_stats)
plt.title('Highest score for each generation')
plt.plot(data[:, 0], data[:, 1], linestyle='--', marker='o')
plt.xlabel('Generation')
plt.ylabel('Highest score')
plt.show(block=True)
