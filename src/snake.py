class Snake():
    # Constructor
    def __init__(self, body_color, head_color, x_pos_initial, y_pos_initial, grid_size, min_x, max_x, min_y, max_y):
        self.body_color = body_color
        self.head_color = head_color

        self.position = [x_pos_initial, y_pos_initial]
        self.tail = []
        self.xbounds = (min_x, max_x)
        self.ybounds = (min_y, max_y)
    
        self.score = 0
        self.grid_size = grid_size
        self.direction = 'RIGHT'
        
    
    def move(self):
        '''Moves the snake in the direction it is currently facing.'''
        if len(self.tail) >= 1:
            self.tail.append((self.position[0], self.position[1]))
            self.tail.pop(0)

        if self.direction == 'UP':
            self.position[1] -= self.grid_size
        elif self.direction == 'DOWN':
            self.position[1] += self.grid_size
        elif self.direction == 'LEFT':
            self.position[0] -= self.grid_size
        elif self.direction == 'RIGHT':
            self.position[0] += self.grid_size

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
        
