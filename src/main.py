import random
import pygame as pg
import time

from snake import Snake
from apple import Apple

# Resolution
WINDOW_SIZE_X = 600
WINDOW_SIZE_Y = WINDOW_SIZE_X

GRID_SIZE = 20

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

border_rect = pg.Rect(0, 0, WINDOW_SIZE_X, WINDOW_SIZE_Y)

# Fonts
SCORE_FONT = 'times new roman'
SCORE_FONT_SIZE = 20
GAME_OVER_FONT = 'times new roman'
GAME_OVER_FONT_SIZE = 25

# Instantiate the snake on the left side of the screen
X_POS_INITIAL = 0
Y_POS_INITIAL = random.randint(0, 29)*GRID_SIZE

snake = Snake(
    body_color=SNAKE_BODY_COLOR,
    head_color=SNAKE_HEAD_COLOR, 
    x_pos_initial=X_POS_INITIAL, 
    y_pos_initial=Y_POS_INITIAL,
    grid_size=GRID_SIZE, 
    min_x=0, 
    max_x=WINDOW_SIZE_X - GRID_SIZE, 
    min_y=0, 
    max_y=WINDOW_SIZE_Y - GRID_SIZE)

# Instantiate the apple
apple = Apple(
    color=APPLE_COLOR, 
        x_pos_initial=0, 
        y_pos_initial=0, 
        grid_size=GRID_SIZE
)

apple.respawn()

# Pygame setup, returns a tuple with the number of successfull and failed inits
n_successful, n_errors  = pg.init()
if n_errors > 0:
    print(f'[!] Encountered {n_errors} error(s) when running pygame.init(), aborting...')
else:
    print('[+] Game initialized successfully!')

pg.display.set_caption("Snakenet")
screen = pg.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
clock = pg.time.Clock()

def showScore(score, color, font, font_size):
    score_font = pg.font.SysFont(font, font_size)

    score_surface = score_font.render('Score: ' + str(score), True, color)

    score_rect = score_surface.get_rect()
    
    # Displays the text
    screen.blit(score_surface, score_rect)

def gameOver(score, color, font, font_size):
    game_over_font = pg.font.SysFont(font, font_size)

    game_over_surface = game_over_font.render('Game over. Final score: ' + str(score), True, color)

    game_over_rect = game_over_surface.get_rect()

    game_over_rect.midtop = (WINDOW_SIZE_X/2, WINDOW_SIZE_Y/4)

    screen.blit(game_over_surface, game_over_rect)
    pg.display.update()

    print('[+] Game over!')

    time.sleep(2)
    pg.quit()

    quit(0)


game_is_running = True
accumulated_time = 0
temp_direction = snake.direction

while game_is_running:
    deltaTime = clock.tick(GAME_FPS) / 1000 # In seconds
    accumulated_time += deltaTime

    logic_time_interval = 1 / SNAKE_MOVES_PER_SECOND # How long should a logical tick be for the current frame

    for event in pg.event.get():
        if event.type == pg.QUIT:
            game_is_running = False

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_UP:
                temp_direction = 'UP'
            elif event.key == pg.K_DOWN:
                temp_direction = 'DOWN'
            elif event.key == pg.K_LEFT:
                temp_direction = 'LEFT'
            elif event.key == pg.K_RIGHT:
                temp_direction = 'RIGHT'

    # Logical tick loop
    while accumulated_time >= logic_time_interval:
        # Check if the player wants to move in the opposite direction of the last movement made by the snake
        # and stop the player from moving the snake inside its own body
        if temp_direction == 'UP' and snake.direction != 'DOWN':
            snake.direction = 'UP'
        elif temp_direction == 'DOWN' and snake.direction != 'UP':
            snake.direction = 'DOWN'
        elif temp_direction == 'LEFT' and snake.direction != 'RIGHT':
            snake.direction = 'LEFT'
        elif temp_direction == 'RIGHT' and snake.direction != 'LEFT':
            snake.direction = 'RIGHT'

        snake.move()

        if snake.isOutOfBounds() or snake.isBitingTail():
            gameOver(
                score=snake.score,
                color=GAME_OVER_TEXT_COLOR,
                font=GAME_OVER_FONT,
                font_size=GAME_OVER_FONT_SIZE
            )

        if snake.position == apple.position:
            snake.score += 1
            snake.grow()
            apple.respawn()

        accumulated_time -= logic_time_interval
        
    # Drawing stuff
    # Clear screen
    screen.fill(BACKGROUND_COLOR)
    pg.draw.rect(surface=screen, color=BORDER_COLOR, rect=border_rect, width=1)

    showScore(
            score=snake.score,
            color=SCORE_TEXT_COLOR,
            font=SCORE_FONT,
            font_size=SCORE_FONT_SIZE
        )

    # Draw apple
    pg.draw.rect(surface=screen, color=APPLE_COLOR, rect=(apple.position[0], apple.position[1], GRID_SIZE, GRID_SIZE))

    # Draw snake
    pg.draw.rect(surface=screen, color=SNAKE_HEAD_COLOR, rect=(snake.position[0], snake.position[1], GRID_SIZE, GRID_SIZE))
    for pos in snake.tail:
        pg.draw.rect(surface=screen, color=SNAKE_BODY_COLOR, rect=(pos[0], pos[1], GRID_SIZE, GRID_SIZE))

    pg.display.update()



