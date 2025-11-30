import random

class Apple():
    def __init__(self, color, x_pos_initial, y_pos_initial, grid_size):
        self.color = color

        self.position = [x_pos_initial, y_pos_initial]

        self.grid_size = grid_size

    def respawn(self):
        '''Respawns the Apple at a random location in the grid.'''
        # The game will run in a 30x30 grid
        self.position = [random.randint(0, 29)*self.grid_size, random.randint(0,24)*self.grid_size]