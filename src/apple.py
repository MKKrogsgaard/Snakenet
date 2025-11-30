import random

class Apple():
    def __init__(self, color, x_pos_initial, y_pos_initial, squares_per_side, grid_size):
        self.color = color
        self.position = [x_pos_initial, y_pos_initial]

        self.squares_per_side = squares_per_side
        self.grid_size = grid_size

    def respawn(self):
        '''Respawns the Apple at a random location in the grid.'''
        self.position = [random.randint(0, self.squares_per_side - 1)*self.grid_size, random.randint(0, self.squares_per_side - 1)*self.grid_size]