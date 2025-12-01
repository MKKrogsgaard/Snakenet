import random

class Apple():
    def __init__(self, color, x_pos_initial, y_pos_initial, squares_per_side, square_size):
        self.color = color
        self.position = [x_pos_initial, y_pos_initial]

        self.squares_per_side = squares_per_side
        self.square_size = square_size

    def respawn(self):
        '''Respawns the Apple at a random location in the grid.'''
        self.position = [random.randint(0, self.squares_per_side - 1)*self.square_size, random.randint(0, self.squares_per_side - 1)*self.square_size]

    def getGridPosition(self):
        '''Returns the position of the apple in grid coordinates'''
        return [int(self.position[0] / self.square_size), int(self.position[1] / self.square_size)]