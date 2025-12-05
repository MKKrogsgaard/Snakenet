from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

from game import Game

def sigmoid(x):
    x = np.clip(x, -500, 500) # To avoid overflow errors
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

alpha = 1.0
def ELU(x):
    x = np.clip(x, -500, 500)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def tanh(x):
    return np.tanh(x)

def identity(x):
    '''Returns x'''
    return x

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
        self.max_ticks_without_eating = 500
        self.tick_counter = 0
        self.ticks_without_eating = 0
        self.total_ticks_survived = 0
    
    def getFitness(self, game_fps, snake_moves_per_second):
        '''Makes the agent play a game of snake and returns the score.'''
        self.tick_counter = 0
        self.total_ticks_survived = 0

        self.game = Game(agent=self, game_fps=game_fps, snake_moves_per_second=snake_moves_per_second)
        apples_eaten, distance_to_apple, distance_to_closest_wall = self.game.startGame() # distance_to_apple is in taxicab distance
        
        self.apples_eaten = apples_eaten
        self.distance_to_apple = distance_to_apple
        self.distance_to_closest_wall = distance_to_closest_wall
        self.total_ticks_survived += self.ticks_without_eating

        score = self.apples_eaten - self.distance_to_apple

        self.fitness = score
        self.ticks_without_eating = 0
        return score



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

        old_top_score = self.best_agent.fitness
        current_top_score = top_agents[0].fitness

        if current_top_score > old_top_score:
            self.best_agent = top_agents[0]
            print(f'[i] Updated best agent. Previous/current best fitness: {old_top_score} / {current_top_score}')

        return top_agents
        

    def crossover(self, agents: List[Agent], layers: List, population_size: int):
        '''
        Performs crossover between selected agents to generate a new population. Returns elites and offspring.

        Works by selecting a parent by coinflip, and picking the current gene from that parent, until all genes are picked.
        '''
        elites = agents # Always include the top agents from the previous generation
        offspring = []
        
        # Each iteration will generate two children from two parents
        iterations_needed = (population_size - len(elites)) // 2

        random_indeces_1 = np.random.randint(0, len(agents), iterations_needed)
        random_indeces_2 = np.random.randint(0, len(agents), iterations_needed)
        for i in tqdm(range(iterations_needed)):
            child1 = agents[random_indeces_1[i]]
            child2 = agents[random_indeces_2[i]]
            
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

        return offspring

    def mutate(self, agents: List[Agent], p: float, std: float):
        '''Mutates the agents by changing a random weight. p is the probability of a mutation occuring for a given weight/bias.'''
        if not(0 <= p <= 1):
            print(f'[!] GeneticAlgorithm.mutate(): p must be in [0,1]. Setting p=1.')
            p = 1

        for i in tqdm(range(len(agents))):
            agent = agents[i]

            for j, weight in enumerate(agent.neural_network.weights):
                mask = np.random.uniform(0, 1, size=weight.shape) < p
            
                mutation_to_apply = np.random.randn(weight.shape[0], weight.shape[1]) * std
                
                agent.neural_network.weights[j] = weight + mutation_to_apply * mask
            
            for j, bias in enumerate(agent.neural_network.biases):
                mask = np.random.uniform(0, 1, size=bias.shape) < p
            
                mutation_to_apply = np.random.randn(bias.shape[0]) * std
                
                agent.neural_network.biases[j] = bias + mutation_to_apply * mask

        return agents
            
    def execute(self, p_selection, p_mutation, std_mutation):
        start_time = time.time()
        self.generation_stats = []

        # Generate intial agents
        self.agents = self.generateAgents(layers=self.layers, population_size=self.population_size)
        self.best_agent = self.agents[0]
        for i in range(self.num_generations):
            print(f'[+] Current generation: {i + 1}')
            print('Selecting agents...')
            elites = self.selectAgents(agents=self.agents, p=p_selection)

            highest_score = max([agent.fitness for agent in self.agents])
            self.generation_stats.append([i + 1, highest_score])

            print('Performing crossover...')
            offspring = self.crossover(agents=elites, layers=self.layers, population_size=self.population_size)

            print('Performing mutation...')
            offspring = self.mutate(agents=offspring, p=p_mutation, std=std_mutation) # Only mutate offspring
            new_population = elites
            new_population.extend(offspring)
            new_population.extend([self.best_agent]) # Always keep the all time best agent
            self.agents = new_population
    
            print(f'[+] Highest fitness of generation {i + 1}: {highest_score}')

        end_time = time.time()
        total_time = end_time - start_time
        print(f'[+] Simulated {self.num_generations} generations of {self.population_size} Agents in {total_time:.2f} seconds at an average of {total_time/self.num_generations:.2f} seconds/generation.')
        
        print(f'The best agent of generation {self.num_generations}:\n  Ate {self.best_agent.apples_eaten} apple(s)\n  Survived for {self.best_agent.total_ticks_survived} logical tick(s)\n  Died at a distance of {self.best_agent.distance_to_apple} from the apple')

        self.best_agent.game.saveGridRecordsToJSON(f'replays/best-agent.json')

        self.best_agent.game.replay(self.snake_moves_per_second, title=f'Best agent of generation {self.num_generations}') # Replay the game from the best agent in the last generation
        
            
LAYERS = [
    [2 + 2 + 1 + 4, 100, ELU],
    [None, 100, ELU],
    [None, 100, ELU],
    [None, 100, ELU],
    [None, 50, ELU],
    [None, 4, identity]
]

POPULATION_SIZE = 1000

NUM_GENERATIONS = 100

ga = GeneticAlgorithm(
    layers=LAYERS,
    population_size=POPULATION_SIZE,
    num_generations=NUM_GENERATIONS,
    game_fps=0, # Uncapped
    snake_moves_per_second=7
)

ga.execute(p_selection=0.05, p_mutation = 0.3, std_mutation=0.5)

data = np.array(ga.generation_stats)
plt.title('Highest score for each generation')
plt.plot(data[:, 0], data[:, 1], linestyle='--', marker='o')
plt.xlabel('Generation')
plt.ylabel('Highest score')
plt.tight_layout()
plt.savefig('generation_stats.png')
