import random

class Apple():
    def __init__(self, color, x_pos_initial, y_pos_initial, squares_per_side, square_size):
        self.color = color
        self.position = [x_pos_initial, y_pos_initial]

        self.squares_per_side = squares_per_side
        self.square_size = square_size

    def respawn(self, position=None):
        '''Respawns the Apple at a given position (pixel coordinates), or at a random position if none is given.'''
        if position != None:
            self.position = [position[0], position[1]]
        else:
            self.position = [random.randint(0, self.squares_per_side - 1)*self.square_size, random.randint(0, self.squares_per_side - 1)*self.square_size]

    def getGridPosition(self, normalize=False):
        '''Returns the position of the apple in grid coordinates'''
        if normalize:
            return [int(self.position[0] / self.square_size) / self.squares_per_side, int(self.position[1] / self.square_size) / self.squares_per_side]

        return [int(self.position[0] / self.square_size), int(self.position[1] / self.square_size)]