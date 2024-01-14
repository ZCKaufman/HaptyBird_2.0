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
        self.gate_width = params["gate_size"] + 1
        self.right_gate = [0, 0, 0] # Left coord, right coord, bot 0 or bot 1
        self.right_gate[0] = np.random.randint((params["game_width"] / 2), (params["game_width"] * 0.95) - self.gate_width)
        self.right_gate[1] = self.right_gate[0] + self.gate_width
        self.right_center = np.floor((self.right_gate[0] + self.right_gate[1]) / 2)
        
        self.left_gate = [0, 0, 0]
        self.left_gate[1] = np.random.randint(self.gate_width, self.right_gate[0] * 0.95)
        self.left_gate[0] = self.left_gate[1] - self.gate_width
        self.left_center = np.floor((self.left_gate[0] + self.left_gate[1]) / 2)

        gate_assignment = random.random()
        if(gate_assignment > 0.5):
            self.right_gate[2] = 1
        else:
            self.left_gate[2] = 1

    def update_y(self, params):
        ### INPUT: params
        ### OUTPUT: Whether or not there was a collision with the cursor(s)
        ### DESCRIPTION: Moves the wall down by 1pt on the y axis and detects if it should be hitting cursors or not
        self.y += 1
        if(self.y == params["cursor_y"]):
            return True
        else:
            return False
