# Snakenet

## Project overview
I made this project to learn more about genetic algorithms. The goal was to create an AI agent that could outperform me (and more importantly, my friends and family) in snake.

It taught me that genetic algorithms suck: They're finnicky, and they take forever to converge to a good solution compared to, say, reinforcement learning. I will not be doing this again in the future :P

The basic concept is this:
1. A large number of AI agents are generated randomly.
2. Each agent plays one or multiple games of snake and is assigned a fitness score based on its performance across those games.
3. The best agents are selected and combined in a crossover process to produce new, similar agents, which are then mutated (i.e. subjected to small random changes).
4. The new population of agents is evaluated and the process repeats.

Each agent recieves input from the game every frame, and uses a neural network to decide which direction the snake should move in for that frame. The architecture of the network is a multi-layer perceptron, but the amount and size of the layers can be customized, as well as the activation function used in each layer.

The fitness function can also be customized. The most successfull fitness function so far combined the number of apples eaten with a small reward for survival time, as well as a reward for each frame where the agent moved toward the apple and a penalty for each frame where it moved away from the apple.

## Agent design

Each agent recieves input from the game every frame, and uses a neural network to decide which direction the snake should move in for that frame. The architecture of the network is a multi-layer perceptron, but the amount and size of the layers can be customized, as well as the activation function used in each layer.

The agent recieves the following input from the game:
- $x$- and $y$-distance from the head of the snake to the apple.
- Distance from the head to each of the four walls.
- Whether positions in a 3x3 grid centered on the head of the snake are occupied by pieces of the tail or not. If the snake is next to a wall, the positions in the 3x3 grid that are out of bounds are counted as occupied.

Since the point in the middle of the 3x3 grid is centered on the head of the snake, it provides no useful information and is thus omitted. So the agent has a total of 2 + 4 + 8 = 14 input neurons.

## Results from experiments

After a lot of tinkering and preliminary runs, I ran 1000 generations of 1000 agents per generation, which produced the agent shown in the video below:

https://www.youtube.com/watch?v=r4p8gV-szDQ

The agent in the video manages to reach a score of 82, which is quite a bit better than me, and decently close to the theoretical maximum score of 100 (the game plays out in a 10x10 grid).

The plots below show the performance of the agents over time. The blue plot shows the fitness of the best agent in the generation, while the orange plot shows the mean number of logical ticks (equivalent to the number of game frames in this case). The plots are normalized to make them easier to interpret.

<img width="1000" height="800" alt="generation_stats" src="https://github.com/user-attachments/assets/a8c85b53-58c6-4e19-ad6b-6843ea141692" />

As mentioned, the fitness functions gave the agents a small reward simply for surviving, but this reward was orders of magnitude lower than those associated with eating apples. It seems that at first the agents developed a strategy that favoured survival over eating apples, judging by the orange graph.

Then around generation 150-200 they changed strategy to focus on eating apples. After that point the fitness graph flattens out, presumably because the agent now had to contend with a new challenge in the form of the longer tails that come with eating more apples.

Performance seems to improve somewhat from generation 600 before settling into a nother plateau around generation 800, presumably due to the constraints imposed by the agents limited perception of the tail. 

Even so, it should be theoretically possible for the agents to achieve a training score with enough time, simply by moving in a fixed pattern that avoids the tail until all 100 apples are eaten.






