from typing import List

import numpy as np
from tqdm import tqdm
import time
import os
import concurrent.futures
import multiprocessing

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

def flattenGenome(agent: 'Agent'):
    '''Returns flattened versions of an agents weights/biases, together with the shapes neede to unflatten them.'''
    weight_shapes = [arr.shape for arr in agent.neural_network.weights]
    bias_shapes = [arr.shape for arr in agent.neural_network.biases]

    weight_genes = np.concatenate([arr.flatten() for arr in agent.neural_network.weights])
    bias_genes = np.concatenate([arr.flatten() for arr in agent.neural_network.biases])

    return weight_genes, weight_shapes, bias_genes, bias_shapes

def task_evaluateGenome(task):
    '''
    Worker function. Must be defined at top level for parallelization.
    
    task is a tuple: (weight_genes, weight_shapes, bias_genes, bias_shapes, layers, game_fps, snake_moves_per_second, num_repeats)

    Returns: (mean_score, apples_eaten, total_ticks_survived, mean_distance_to_apple, game)
    '''

    # Limits how many threads numerical libraries can use, to (hopefully) leave enough free threads for the fitness evaluation
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    weight_genes, weight_shapes, bias_genes, bias_shapes, layers, game_fps, snake_moves_per_second, num_repeats = task

    # Reconstruct the agent
    agent = Agent(layers=layers)
    agent.neural_network.weights = unflatten(weight_genes, weight_shapes)
    agent.neural_network.biases = unflatten(bias_genes, bias_shapes)

    scores = []
    for i in range(num_repeats):
        temp_score = agent.getFitness(game_fps=game_fps, snake_moves_per_second=snake_moves_per_second)
        scores.append(temp_score)
    
    mean_score = np.mean(scores)
    apples_eaten = agent.apples_eaten
    total_ticks_survived = agent.total_ticks_survived
    mean_distance_to_apple = agent.mean_distance_to_apple

    return mean_score, apples_eaten, total_ticks_survived, mean_distance_to_apple


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
        self.max_ticks_without_eating = 100
        self.ticks_without_eating = 0
        self.total_ticks_survived = 0
    
    def getFitness(self, game_fps, snake_moves_per_second):
        '''Makes the agent play a game of snake and returns the score.'''
        self.total_ticks_survived = 0

        self.game = Game(agent=self, game_fps=game_fps, snake_moves_per_second=snake_moves_per_second)
        apples_eaten, final_distance_to_apple, final_distance_to_closest_wall, mean_distance_to_apple = self.game.startGame() # distance_to_apple is in taxicab distance
        
        self.apples_eaten = apples_eaten
        self.final_distance_to_apple = final_distance_to_apple
        self.mean_distance_to_apple = mean_distance_to_apple
        self.final_distance_to_closest_wall = final_distance_to_closest_wall

        score = 10*self.apples_eaten + 0.01*self.total_ticks_survived - 10*self.mean_distance_to_apple

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

        self.best_historical_score = 0
        self.best_game = None

    def generateAgents(self, layers, population_size):
        return [Agent(layers=layers) for i in range(population_size)]

    def getAgentFitnessScores(self, agents: List[Agent], num_repeats_per_agent):
        try:
            multiprocessing.set_start_method('spawn', force=False)
        except RuntimeError:
            # Start method has already been set, do nothing
            pass

        # Set number of workers to available CPU cores (minus 1 for the OS)
        max_workers = max(1, os.cpu_count() - 1)

        tasks = []
        for agent in agents:
            weight_genes, weight_shapes, bias_genes, bias_shapes = flattenGenome(agent)
            tasks.append((weight_genes, weight_shapes, bias_genes, bias_shapes, self.layers, self.game_fps, self.snake_moves_per_second, num_repeats_per_agent))
        
        results = [None for i in range(len(agents))]

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task_evaluateGenome, task) for task in tasks] # Represents the results of the scheduled CPU calls
            for i in tqdm(range(len(futures))):
                mean_score, apples_eaten, total_ticks_survived, mean_distance_to_apple = futures[i].result()
                
                agents[i].fitness = mean_score
                agents[i].apples_eaten = apples_eaten
                agents[i].total_ticks_survived = total_ticks_survived
                agents[i].mean_distance_to_apple = mean_distance_to_apple

                results[i] = mean_score

        return results
    
    def selectAgents(self, agents: List[Agent], p: int):
        '''
        Selects the agents with the highest fitness score from the list of agents.
        
        :p: The percentile of agents to select. For example, p=0.1 means we select only the top 10 % of agents.
        '''

        if p > 1:
            print(f'[!] GeneticAlgorithm.selectAgents(): p cannot be greater than 1. Setting p=1.')
            p = 1
        
        scores = self.getAgentFitnessScores(agents=agents, num_repeats_per_agent=10)

        agent_score_pairs = zip(agents, scores)

        sorted_pairs = sorted(agent_score_pairs, key=lambda pair: pair[1], reverse=True)
        sorted_agents = [pair[0] for pair in sorted_pairs]

        index = max(1, int(p * len(agents)))
        top_agents = sorted_agents[0:index]

        old_best_agent_score = self.best_agent.fitness
        current_top_score = top_agents[0].fitness

        if current_top_score > old_best_agent_score:
            self.best_agent = top_agents[0]
            # Save a replay of the best agent
            self.best_agent.game = Game(agent=self.best_agent, game_fps=self.game_fps, snake_moves_per_second=self.snake_moves_per_second)
            self.best_agent.game.startGame()
            self.best_agent.game.saveGridRecordsToJSON(f'replays/best-agent.json')
        else:
            top_agents[0] = self.best_agent

        return top_agents
        
    def crossover(self, agents: List[Agent], layers: List, population_size: int):
        '''
        Performs crossover between selected agents to generate a new population. Returns elites and offspring.

        Works by selecting a parent by coinflip, and picking the current gene from that parent, until all genes are picked.
        '''
        offspring = []
        
        # Each iteration will generate two children from two parents
        iterations_needed = (population_size - len(agents)) // 2

        random_indeces_1 = np.random.randint(0, len(agents), iterations_needed)
        random_indeces_2 = np.random.randint(0, len(agents), iterations_needed)
        for i in tqdm(range(iterations_needed)):
            parent1 = agents[random_indeces_1[i]]
            parent2 = agents[random_indeces_2[i]]
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

        self.best_historical_score = 0
        for i in range(self.num_generations):
            print(f'[+] Current generation: {i + 1}')
            print('Selecting agents...')
            elites = self.selectAgents(agents=self.agents, p=p_selection)

            highest_current_score = max([agent.fitness for agent in self.agents])
            print(f'[+] Highest average fitness of generation {i + 1}: {highest_current_score}')

            # If the highest score from this generation exceeds the highest all-time score, play the replay
            self.generation_stats.append([i + 1, highest_current_score])
            if self.best_historical_score < highest_current_score:
                self.best_agent.game.loadGridRecordsFromJSON(f'replays/best-agent.json')
                self.best_agent.game.replay(snake_moves_per_second=self.snake_moves_per_second, title=f'Current best snake (generation {i + 1})')

            print('Performing crossover...')
            offspring = self.crossover(agents=elites, layers=self.layers, population_size=self.population_size)

            print('Performing mutation...')
            offspring = self.mutate(agents=offspring, p=p_mutation, std=std_mutation) # Only mutate offspring
            new_population = elites
            new_population.extend(offspring)

            self.agents = new_population

            


        end_time = time.time()
        total_time = end_time - start_time
        print(f'[+] Simulated {self.num_generations} generations of {self.population_size} Agents in {total_time:.2f} seconds at an average of {total_time/self.num_generations:.2f} seconds/generation.')
        
        print(f'The best agent of generation {self.num_generations}:\n  Ate {self.best_agent.apples_eaten} apple(s)\n  Survived for {self.best_agent.total_ticks_survived} logical tick(s)\n  With mean distance to appple {self.best_agent.mean_distance_to_apple} from the apple')

        

