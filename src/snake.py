# Type hinting for classes that aren't actually imported, courtesey of StackOverflow 
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from apple import Apple
    from game import Grid

class Snake():
    # Constructor
    def __init__(self, body_color, head_color, x_pos_initial, y_pos_initial, square_size, squares_per_side, min_x, max_x, min_y, max_y):
        self.body_color = body_color
        self.head_color = head_color

        self.position = [x_pos_initial, y_pos_initial]
        self.tail = []
        self.xbounds = (min_x, max_x)
        self.ybounds = (min_y, max_y)
    
        self.score = 0
        self.square_size = square_size
        self.squares_per_side = squares_per_side

        self.direction = 'RIGHT'
        
    
    def move(self):
        '''Moves the snake in the direction it is currently facing.'''
        # Don't append a piece to the tail if the current position is out of bounds!
        if not self.isOutOfBounds():
            if len(self.tail) >= 1:
                self.tail.append((self.position[0], self.position[1]))
                self.tail.pop(0)

        if self.direction == 'UP':
            self.position[1] -= self.square_size
        elif self.direction == 'DOWN':
            self.position[1] += self.square_size
        elif self.direction == 'LEFT':
            self.position[0] -= self.square_size
        elif self.direction == 'RIGHT':
            self.position[0] += self.square_size

    def grow(self):
        '''Grows the snake by appending the current position of the head to its tail.'''
        self.tail.append((self.position[0], self.position[1]))

    def isOutOfBounds(self):
        if (self.xbounds[0] <= self.position[0] <= self.xbounds[1]) and (self.ybounds[0] <= self.position[1] <= self.ybounds[1]):
            return False
        else:
            return True
        
    def isBitingTail(self):
        for pos in self.tail:
            if self.position[0] == pos[0] and self.position[1] == pos[1]:
                return True
        return False
    
    def getHeadGridPosition(self, normalize=False):
        '''Returns the position of the head in grid coordinates.'''
        if normalize:
            return [int(self.position[0] / self.square_size*self.squares_per_side), int(self.position[1] / self.square_size*self.squares_per_side)]
        
        return [int(self.position[0] / self.square_size), int(self.position[1] / self.square_size)]
        
    def getTailGridPositions(self):
        '''Returns the positions of the tail elements in grid coordinates.'''
        res = []
        for pos in self.tail:
            res.append([int(pos[0] / self.square_size), int(pos[1] / self.square_size)])
        return res

    def getDistanceToApple(self, apple: Apple, normalize=False):
        '''
        Returns the taxicab distance from the head of the snake to the apple, as well as x and y distance.
        '''
        x_distance = self.position[0] - apple.position[0]
        y_distance = self.position[1] - apple.position[1]

        total_distance = abs(self.position[0] - apple.position[0]) + abs(self.position[1] - apple.position[1])

        if normalize == True:
            x_distance = x_distance / (self.square_size*self.squares_per_side)
            y_distance = y_distance / (self.square_size*self.squares_per_side)
            total_distance = total_distance / (2*self.square_size*self.squares_per_side)

        return total_distance, x_distance, y_distance

    def getDistanceToWalls(self, normalize=False):
        left_wall_distance = (self.position[0] - self.xbounds[0])
        right_wall_distance = (self.xbounds[1] - self.position[0])
        bottom_wall_distance = (self.position[1] - self.ybounds[0])
        top_wall_distance = (self.ybounds[1] - self.position[1])

        if normalize:
            return [left_wall_distance/(self.square_size * self.squares_per_side), right_wall_distance/(self.square_size * self.squares_per_side), top_wall_distance/(self.square_size * self.squares_per_side), bottom_wall_distance/(self.square_size * self.squares_per_side)]

        return [left_wall_distance, right_wall_distance, top_wall_distance, bottom_wall_distance]
    
    def lookForTail(self, grid: 'Grid'):
        '''
        Looks for pieces of the tail in a 3x3 grid around the snake's head and returns an array of 8 binary values showing if a space is occupied or not (the center is always occupied by the snake's head, so no need to return a value for that).

        If the snake is near the wall, the out of bounds squares in the grid are counted as occupied. The notation used for the positions in the grid around the head is shown below.

        11  12  13
        21  HH  23
        31  32  33   
        '''
        head_x, head_y = self.getHeadGridPosition()
        
        positions = [[None for i in range(3)] for i in range(3)]
        positions[1][1] = 0 # The head

        # x is out of bounds left
        if head_x - 1 < 0:
            positions[0][0] = 1
            positions[1][0] = 1
            positions[2][0] = 1
        # x is out of bounds right
        if head_x + 1 >= self.squares_per_side:
            positions[0][2] = 1
            positions[1][2] = 1
            positions[2][2] = 1
        # y is out of bounds down
        if head_y - 1 < 0:
            positions[2][0] = 1
            positions[2][1] = 1
            positions[2][2] = 1
        # y is out of bounds up
        if head_y + 1 >= self.squares_per_side:
            positions[0][0] = 1
            positions[0][1] = 1
            positions[0][2] = 1

        for i in range(len(positions)):
            for j in range(len(positions)):
                if positions[i][j] == None:
                    if (0 <= head_x - 1 + i < self.squares_per_side) and (0 <= head_y - 1 + j < self.squares_per_side):
                        positions[i][j] = grid.body_channel[head_x - 1 + i][head_y - 1 + j] # See https://www.baeldung.com/cs/finding-neighbors-of-matrix-element
                    else:
                        positions[i][j] = 1

        return positions
