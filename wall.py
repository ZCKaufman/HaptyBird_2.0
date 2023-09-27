import numpy as np
import random
import sys
import pygame

class Wall:
    def __init__(self, params): # Gate width is currently hard coded to 3
        ### INPUT: params
        ### OUTPUT: None
        ### DESCRIPTION: Generates a wall the size of the game at top of screen, and a positive gate within that wall
        self.y = 0 
        self.gate_width = 10
        self.left = np.random.randint(params["game_width"] * 0.05, (params["game_width"] * 0.95) - self.gate_width)
        self.right = self.left + self.gate_width

    def update_y(self, params):
        ### INPUT: params
        ### OUTPUT: Whether or not there was a collision with the cursor(s)
        ### DESCRIPTION: Moves the wall down by 1pt on the y axis and detects if it should be hitting cursors or not
        self.y += 1
        if(self.y == params["cursor_y"]):
            return True
        else:
            return False
