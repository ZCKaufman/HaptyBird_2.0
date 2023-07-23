import numpy as np
import random
from random import randint

class Player:
    def __init__(self, params):
        self.score = 0
        self.x = params["player_start_x"]
        self.y = params["cursor_y_axis"]
        self.x_change = -1 # Arbitrarily start by going left

    def move(self, params):
        # Will eventually connect to the logitech
        new_x = randint(0, params["game_width"])
        self.x_change = self.x - new_x
        self.x = new_x
        '''if(self.x < 300):
            self.x += 1
            self.x_change = 1
        else:
            self.x -= 1
            self.x_change = -1'''