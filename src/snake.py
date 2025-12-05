# Type hinting for classes that aren't actually imported, courtesey of StackOverflow 
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from apple import Apple

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
        if not(self.xbounds[0] <= self.position[0] <= self.xbounds[1] and self.ybounds[0] <= self.position[1] <= self.ybounds[1]):
            return True
        else:
            return False
        
    def isBitingTail(self):
        for pos in self.tail:
            if self.position[0] == pos[0] and self.position[1] == pos[1]:
                return True
        return False
    
    def getHeadGridPosition(self, normalize=False):
        '''Returns the position of the head in grid coordinates.'''
        if normalize:
            return [int(self.position[0] / self.square_size) / self.squares_per_side, int(self.position[1] / self.square_size)  / self.squares_per_side]
        
        return [int(self.position[0] / self.square_size), int(self.position[1] / self.square_size)]
        
    def getTailGridPositions(self):
        '''Returns the positions of the tail elements in grid coordinates.'''
        res = []
        for pos in self.tail:
            res.append([int(pos[0] / self.square_size), int(pos[1] / self.square_size)])
        return res

    def getDistanceToApple(self, apple: Apple, normalize=False):
        '''Returns the taxicab distance from the head of the snake to the apple.'''
        dist = abs(self.position[0] - apple.position[0]) + abs(self.position[1] - apple.position[1])

        if normalize == True:
            dist = dist / (2*self.square_size*self.squares_per_side)

        return dist 

    def getDistanceToWalls(self, normalize=False):
        left_wall_distance = (self.position[0] - self.xbounds[0])
        right_wall_distance = (self.xbounds[1] - self.position[0])
        bottom_wall_distance = (self.position[1] - self.ybounds[0])
        top_wall_distance = (self.ybounds[1] - self.position[1])

        if normalize:
            return [left_wall_distance/(self.square_size * self.squares_per_side), right_wall_distance/(self.square_size * self.squares_per_side), top_wall_distance/(self.square_size * self.squares_per_side), bottom_wall_distance/(self.square_size * self.squares_per_side)]

        return [left_wall_distance, right_wall_distance, top_wall_distance, bottom_wall_distance]

