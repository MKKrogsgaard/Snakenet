# Type hinting for classes that aren't actually imported, courtesey of StackOverflow 
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geneticalgorithm import Agent

import random
import pygame as pg
import time
import numpy as np

from snake import Snake
from apple import Apple

def softmax(arr):
    arr = np.array(arr)

    return np.exp(arr) / np.sum(np.exp(arr))

# Resolution
WINDOW_SIZE = 500

SQUARES_PER_SIDE = 20 # Grid dimensions will be SQUARES_PER_SIDE^2
SQUARE_SIZE = WINDOW_SIZE / SQUARES_PER_SIDE

# FPS and game speed
GAME_FPS = 60
SNAKE_MOVES_PER_SECOND = 6

# Colors as pygame color objects instead of (R, G, B)
BACKGROUND_COLOR = pg.Color(0, 0, 0)
SCORE_TEXT_COLOR = pg.Color(255, 255, 255)
GAME_OVER_TEXT_COLOR = pg.Color(255, 0, 0)
BORDER_COLOR = pg.Color(255, 255, 255)
SNAKE_BODY_COLOR = pg.Color(31, 158, 9)
SNAKE_HEAD_COLOR = pg.Color(57, 252, 22)
APPLE_COLOR = pg.Color(252, 22, 38)

border_rect = pg.Rect(0, 0, WINDOW_SIZE, WINDOW_SIZE)

# Fonts
SCORE_FONT = 'times new roman'
SCORE_FONT_SIZE = 20
GAME_OVER_FONT = 'times new roman'
GAME_OVER_FONT_SIZE = 22

# Instantiate the snake on the left side of the screen
X_POS_INITIAL = 0
Y_POS_INITIAL = random.randint(0, SQUARES_PER_SIDE - 1)*SQUARE_SIZE
# Y_POS_INITIAL = 0

# For grid positions
APPLE_VAL = 1
BODY_VAL = -1
HEAD_VAL = 0.5
EMPTY_SQUARE_VAL = 0

class Grid():
    '''
    Keeps track of the positions of objects in the game grid via the attribute 'position'.
    
    position[i][j] is the object that currently occupies the square at grid coordinate (i, j).
    '''
    def __init__(self, squares_per_side, square_size):
        self.squares_per_side = squares_per_side
        self.square_size = square_size
        
        self.positions = [[EMPTY_SQUARE_VAL for i in range(self.squares_per_side)] for i in range(self.squares_per_side)]

    def update(self, snake=Snake, apple=Apple):
        self.positions = [[EMPTY_SQUARE_VAL for i in range(self.squares_per_side)] for i in range(self.squares_per_side)]

        apple_x, apple_y = apple.getGridPosition()
        self.positions[apple_x][apple_y] = APPLE_VAL

        head_x, head_y = snake.getHeadGridPosition()
        self.positions[head_x][head_y] = HEAD_VAL
        
        tailpositions = snake.getTailGridPositions()
        for pos in tailpositions:
            pos_x, pos_y = pos
            self.positions[pos_x][pos_y] = BODY_VAL

    def printGrid(self):
        temp_str = ""
        # The arrangement is self.positions[x][y] with x = column, y = row
        for j in range(self.squares_per_side):
            for i in range(self.squares_per_side):
                temp_str += str(self.positions[i][j])
            temp_str += "\n"
        print(temp_str)

class Game():
    '''Main class that handles all game logic.'''
    def __init__(self, agent: Agent, game_fps:int, snake_moves_per_second:int):
        self.game_fps = game_fps
        self.snake_moves_per_second = snake_moves_per_second
        self.agent = agent

        self.grid_records = []

        self.snake = Snake(
            body_color=SNAKE_BODY_COLOR,
            head_color=SNAKE_HEAD_COLOR, 
            x_pos_initial=X_POS_INITIAL, 
            y_pos_initial=Y_POS_INITIAL,
            square_size=SQUARE_SIZE, 
            min_x=0, 
            max_x=WINDOW_SIZE - SQUARE_SIZE, 
            min_y=0, 
            max_y=WINDOW_SIZE - SQUARE_SIZE)

        self.apple = Apple(
            color=APPLE_COLOR, 
            x_pos_initial=0, 
            y_pos_initial=0,
            squares_per_side=SQUARES_PER_SIDE,
            square_size=SQUARE_SIZE
        )
        self.apple.respawn()

        self.grid = Grid(
            square_size=SQUARE_SIZE,
            squares_per_side=SQUARES_PER_SIDE
        )


    def showScore(self, score, color, font, font_size):
        score_font = pg.font.SysFont(font, font_size)

        score_surface = score_font.render('Score: ' + str(score), True, color)

        score_rect = score_surface.get_rect()
        
        # Displays the text
        self.screen.blit(score_surface, score_rect)

    def gameOver(self, score, color, font, font_size, message):
        if self.agent == None:
            # This is only for human players
            game_over_font = pg.font.SysFont(font, font_size)

            game_over_surface = game_over_font.render(message + str(score), True, color)

            game_over_rect = game_over_surface.get_rect()

            game_over_rect.midtop = (WINDOW_SIZE/2, WINDOW_SIZE/4)

            # Clear screen
            self.screen.fill(BACKGROUND_COLOR)
            pg.draw.rect(surface=self.screen, color=BORDER_COLOR, rect=border_rect, width=1)

            self.screen.blit(game_over_surface, game_over_rect)
            pg.display.update()

            print('[+] Game over!')

            time.sleep(3)
        self.final_score = score
        self.game_is_running = False # Exit the main loop cleanly

    def render(self, grid_positions, score: int):
        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)
        pg.draw.rect(surface=self.screen, color=BORDER_COLOR, rect=border_rect, width=1)

        for i in range(len(grid_positions)):
            for j in range(len(grid_positions)):
                current_position = grid_positions[i][j]
                if current_position == APPLE_VAL:
                    pg.draw.rect(surface=self.screen, color=APPLE_COLOR, rect=(i*SQUARE_SIZE, j*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                elif current_position == BODY_VAL:
                    pg.draw.rect(surface=self.screen, color=SNAKE_BODY_COLOR, rect=(i*SQUARE_SIZE, j*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                elif current_position == HEAD_VAL:
                    pg.draw.rect(surface=self.screen, color=SNAKE_HEAD_COLOR, rect=(i*SQUARE_SIZE, j*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

        self.showScore(
                score=score,
                color=SCORE_TEXT_COLOR,
                font=SCORE_FONT,
                font_size=SCORE_FONT_SIZE
            )

        pg.display.update()

    def replay(self, snake_moves_per_second, title='Snake replay'):
        # Pygame setup, returns a tuple with the number of successfull and failed inits
        n_successful, n_errors  = pg.init()
        if n_errors > 0:
            print(f'[!] Encountered {n_errors} error(s) when initializing the replay, aborting...')
        else:
            print('[+] Replay initialized successfully!')

        fps = snake_moves_per_second # FPS should match how fast the snake moved

        self.screen = pg.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pg.display.set_caption(title)

        self.clock = pg.time.Clock()
        for grid_state in self.grid_records:
            self.render(grid_state[0], grid_state[1])
            self.clock.tick(fps)
        time.sleep(3)
        

    def processInput(self):
        if self.agent == None:
            # Human player
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.game_is_running = False

                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_UP:
                        self.temp_direction = 'UP'
                    elif event.key == pg.K_DOWN:
                        self.temp_direction = 'DOWN'
                    elif event.key == pg.K_LEFT:
                        self.temp_direction = 'LEFT'
                    elif event.key == pg.K_RIGHT:
                        self.temp_direction = 'RIGHT'
        else:
            agent_input = np.array(self.grid.positions)
            agent_input = agent_input.flatten()
            agent_input = np.append(agent_input, self.snake.getDistanceToApple(self.apple, normalize=True, square_size=SQUARE_SIZE, squares_per_side=SQUARES_PER_SIDE))

            agent_output = self.agent.neural_network.forward(agent_input)
            agent_output_softmaxed = softmax(agent_output)

            agent_output_direction = np.random.choice([0, 1, 2, 3], p=agent_output_softmaxed)

            # Press the key corresponding to the output neuron with the greatest activation
            if agent_output_direction == 0:
                self.temp_direction = 'UP'
            elif agent_output_direction == 1:
                self.temp_direction = 'DOWN'
            elif agent_output_direction == 2:
                self.temp_direction = 'LEFT'
            elif agent_output_direction == 3:
                self.temp_direction = 'RIGHT'

    def update(self):
        # Check if the player wants to move in the opposite direction of the last movement made by the snake
        # and stop the player from moving the snake inside its own body
        if self.temp_direction == 'UP' and self.snake.direction != 'DOWN':
            self.snake.direction = 'UP'
        elif self.temp_direction == 'DOWN' and self.snake.direction != 'UP':
            self.snake.direction = 'DOWN'
        elif self.temp_direction == 'LEFT' and self.snake.direction != 'RIGHT':
            self.snake.direction = 'LEFT'
        elif self.temp_direction == 'RIGHT' and self.snake.direction != 'LEFT':
            self.snake.direction = 'RIGHT'

        self.snake.move()

        if self.snake.isOutOfBounds():
            self.gameOver(
                score=self.snake.score,
                color=GAME_OVER_TEXT_COLOR,
                font=GAME_OVER_FONT,
                font_size=GAME_OVER_FONT_SIZE,
                message='Game over, you hit a wall! Final score: '
            )

        if self.snake.isBitingTail():
            self.gameOver(
                score=self.snake.score,
                color=GAME_OVER_TEXT_COLOR,
                font=GAME_OVER_FONT,
                font_size=GAME_OVER_FONT_SIZE,
                message='Game over, you ate your own tail! Final score: '
            )

        if self.snake.position == self.apple.position:
            self.snake.score += 1
            self.snake.grow()
            self.apple.respawn()
            if self.agent != None:
                # Reset iterations counter
                self.agent.iteration_counter = 0
        if self.agent == None:
            self.accumulated_time -= self.logic_time_interval

    def startGame(self):
        # If a human is playing
        if self.agent == None:
            # Pygame setup, returns a tuple with the number of successfull and failed inits
            n_successful, n_errors  = pg.init()
            if n_errors > 0:
                print(f'[!] Encountered {n_errors} error(s) when initializing the game, aborting...')
            else:
                print('[+] Game initialized successfully!')

            self.screen = pg.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pg.display.set_caption('Snake')

        self.clock = pg.time.Clock()

        self.game_is_running = True
        self.accumulated_time = 0
        self.temp_direction = self.snake.direction

        # Agent loop
        if self.agent != None:
            while self.game_is_running:
                # Prevents agents from running in circles to stave off their inevitable doom
                if self.agent.tick_counter > self.agent.max_ticks:
                    self.gameOver(
                        score=self.snake.score,
                        color=GAME_OVER_TEXT_COLOR,
                        font=GAME_OVER_FONT,
                        font_size=GAME_OVER_FONT_SIZE,
                        message='Too many iterations! Final score: '
                    )
                
                self.grid.update(self.snake, self.apple)
                self.processInput()
                self.update()
                self.grid_records.append([self.grid.positions, self.snake.score])
                if not self.game_is_running:
                    break
            
                self.agent.tick_counter += 1
                self.agent.ticks_survived += 1

        # Human loop
        else:
            while self.game_is_running:
                deltaTime = self.clock.tick(self.game_fps) / 1000 # In seconds
                self.accumulated_time += deltaTime

                self.logic_time_interval = 1 / self.snake_moves_per_second # How long should a logical tick be for the current frame

                # Process input every frame, but update movement and render for every logical tick
                self.processInput()
                while self.accumulated_time >= self.logic_time_interval:
                    self.grid.update(self.snake, self.apple)
                    self.update()
                    self.grid_records.append([self.grid.positions, self.snake.score])
                    self.render(self.grid.positions, self.snake.score)
                    if not self.game_is_running:
                        break
                
        pg.quit()

        final_score = self.snake.score
        final_distance_to_apple = self.snake.getDistanceToApple(self.apple)

        return final_score, final_distance_to_apple

# game = Game(
#     agent=None,
#     game_fps=GAME_FPS,
#     snake_moves_per_second=SNAKE_MOVES_PER_SECOND
#     )

# final_score, distance_to_apple = game.startGame()

# game.replay(snake_moves_per_second=SNAKE_MOVES_PER_SECOND)

