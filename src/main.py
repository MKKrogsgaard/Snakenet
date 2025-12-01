import random
import pygame as pg
import time

from snake import Snake
from apple import Apple

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
GAME_OVER_FONT_SIZE = 25

# Instantiate the snake on the left side of the screen
X_POS_INITIAL = 0
Y_POS_INITIAL = random.randint(0, SQUARES_PER_SIDE - 1)*SQUARE_SIZE

class Grid():
    '''
    Keeps track of the positions of objects in the game grid via the attribute 'position'.
    
    position[i][j] is the object that currently occupies the square at grid coordinate (i, j).
    
    Legend:
        0: Unoccupied
        1: Apple
        2: Snake body
        3: Snake head
    '''
    def __init__(self, squares_per_side, square_size):
        self.squares_per_side = squares_per_side
        self.square_size = square_size
        
        self.positions = [[0 for i in range(self.squares_per_side)] for i in range(self.squares_per_side)]

    def update(self, snake=Snake, apple=Apple):
        self.positions = [[0 for i in range(self.squares_per_side)] for i in range(self.squares_per_side)]

        apple_x, apple_y = apple.getGridPosition()
        self.positions[apple_x][apple_y] = 1

        head_x, head_y = snake.getHeadGridPosition()
        self.positions[head_x][head_y] = 3
        
        tailpositions = snake.getTailGridPositions()
        for pos in tailpositions:
            pos_x, pos_y = pos
            self.positions[pos_x][pos_y] = 2

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
    def __init__(self):
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

        # Instantiate the apple
        self.apple = Apple(
            color=APPLE_COLOR, 
            x_pos_initial=0, 
            y_pos_initial=0,
            squares_per_side=SQUARES_PER_SIDE,
            square_size=SQUARE_SIZE
        )
        self.apple.respawn()


    def showScore(self, score, color, font, font_size):
        score_font = pg.font.SysFont(font, font_size)

        score_surface = score_font.render('Score: ' + str(score), True, color)

        score_rect = score_surface.get_rect()
        
        # Displays the text
        self.screen.blit(score_surface, score_rect)

    def gameOver(self, score, color, font, font_size):
        game_over_font = pg.font.SysFont(font, font_size)

        game_over_surface = game_over_font.render('Game over. Final score: ' + str(score), True, color)

        game_over_rect = game_over_surface.get_rect()

        game_over_rect.midtop = (WINDOW_SIZE/2, WINDOW_SIZE/4)

        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)
        pg.draw.rect(surface=self.screen, color=BORDER_COLOR, rect=border_rect, width=1)

        self.screen.blit(game_over_surface, game_over_rect)
        pg.display.update()

        print('[+] Game over!')

        time.sleep(2)
        self.final_score = score
        self.game_is_running = False # Exit the main loop cleanly

    def render(self):
        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)
        pg.draw.rect(surface=self.screen, color=BORDER_COLOR, rect=border_rect, width=1)

        # Draw apple
        pg.draw.rect(surface=self.screen, color=APPLE_COLOR, rect=(self.apple.position[0], self.apple.position[1], SQUARE_SIZE, SQUARE_SIZE))

        # Draw snake
        pg.draw.rect(surface=self.screen, color=SNAKE_HEAD_COLOR, rect=(self.snake.position[0], self.snake.position[1], SQUARE_SIZE, SQUARE_SIZE))
        for pos in self.snake.tail:
            pg.draw.rect(surface=self.screen, color=SNAKE_BODY_COLOR, rect=(pos[0], pos[1], SQUARE_SIZE, SQUARE_SIZE))

        self.showScore(
                score=self.snake.score,
                color=SCORE_TEXT_COLOR,
                font=SCORE_FONT,
                font_size=SCORE_FONT_SIZE
            )

        pg.display.update()

    def processInput(self):
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

        if self.snake.isOutOfBounds() or self.snake.isBitingTail():
            self.gameOver(
                score=self.snake.score,
                color=GAME_OVER_TEXT_COLOR,
                font=GAME_OVER_FONT,
                font_size=GAME_OVER_FONT_SIZE
            )

        if self.snake.position == self.apple.position:
            self.snake.score += 1
            self.snake.grow()
            self.apple.respawn()

        self.accumulated_time -= self.logic_time_interval

    def startGame(self):
        # Pygame setup, returns a tuple with the number of successfull and failed inits
        n_successful, n_errors  = pg.init()
        if n_errors > 0:
            print(f'[!] Encountered {n_errors} error(s) when running pygame.init(), aborting...')
        else:
            print('[+] Game initialized successfully!')

        self.screen = pg.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pg.display.set_caption('Snake')
        self.clock = pg.time.Clock()

        self.game_is_running = True
        self.accumulated_time = 0
        self.temp_direction = self.snake.direction

        while self.game_is_running:
            deltaTime = self.clock.tick(GAME_FPS) / 1000 # In seconds
            self.accumulated_time += deltaTime

            self.logic_time_interval = 1 / SNAKE_MOVES_PER_SECOND # How long should a logical tick be for the current frame

            # Process input every frame, but update movement and render for every logical tick
            self.processInput()
            while self.accumulated_time >= self.logic_time_interval:
                self.update()
                grid.update(self.snake, self.apple)
                self.render()
                grid.printGrid()

        pg.quit()
        return self.snake.score

game = Game()
grid = Grid(
    square_size=SQUARE_SIZE,
    squares_per_side=SQUARES_PER_SIDE
)

final_score = game.startGame()

print(f'Final score: {final_score}')
