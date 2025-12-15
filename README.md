# Snakenet
I made this project to learn more about genetic algorithms. The goal was to create an AI agent that could outperform me (and more importantly, my friends and family) in snake.

The basic concept is this:
1. A large number of AI agents are generated randomly.
2. Each agent plays one or multiple games of snake and is assigned a fitness score based on its performance across those games.
3. The best agents are selected and combined in a crossover process to produce new, similar agents, which are then mutated (i.e. subjected to small random changes).
4. The new population of agents is evaluated and the process repeats.

Each agent recieves input from the game every frame, and uses a neural network to decide which direction the snake should move in for that frame. The architecture of the network is a multi-layer perceptron, but the amount and size of the layers can be customized, as well as the activation function used in each layer.

The fitness function can also be customized. The most successfull fitness function so far combined the number of apples eaten with a small reward for survival time, as well as a reward for each frame where the agent moved toward the apple and a penalty for each frame where it moved away from the apple, i.e.

