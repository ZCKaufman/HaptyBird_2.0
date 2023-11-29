import pygame
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from random import randint

class HaptyBaby():
    def __init__(self, params, first_gen = True, mutant = False, child = False, parent1 = None, parent2 = None):
        ### INPUT: Params
        ### OUTPUT: None
        ### DESCRIPTION: Generates a baby bot. Could be a first generation, mutant, or child

        # GENETIC VARIABLES #
        self.alive = True
        self.fitness = 0
        self.best = False 
        self.input_weights = []
        self.chromosome = []
        self.first_gen = not (mutant or child)
        self.mutant = mutant # One parent (self)
        self.child = child # Two parents
        self.parent1 = parent1
        self.parent2 = parent2

        # GAME RELATED VARIABLES #
        self.x = np.floor(params["game_width"] / 2)
        self.y = params["cursor_y"]
        self.x_change = 0 # Arbitrarily begin by staying still
        self.prev_move = 0
        self.dist_left = 0
        self.dist_right = 0
        self.dist_y = 0
        self.passed = 0

        # GAME STATISTICS #
        self.score = 0

        self.set_weights(params)

    def update_state(self, wall):
        self.dist_left = wall.left - self.x
        self.dist_right = wall.right - self.x
        self.dist_y = self.y - wall.y
        self.fitness += 0.01

    def collision(self, wall):
        if self.x > wall.left and self.x < wall.right:
            self.score += 1
            self.fitness += 3
            self.passed += 1
        else:
            self.score -= 1
            self.fitness -= 3
            self.alive = False

    def move(self, params):
        # Set the inputs to state
        state = [self.x, self.dist_left, self.dist_right, self.dist_y, self.prev_move]

        layer1 = np.dot(state, self.input_weights)
        layer2 = self.sigmoid(layer1)
        layer3 = np.dot(layer2, self.chromosome)
        prediction = self.sigmoid(layer3)

        if (prediction < 0.33 and self.x > 0):
            self.x -= 1
            self.prev_move = -1
        elif (prediction > 0.66 and self.x < params["game_width"]):
            self.x += 1
            self.prev_move = 1
        else:
            self.x = self.x
            self.prev_move = 0

    def set_weights(self, params):
        if (self.first_gen):
            ### RANDOM WEIGHTS ###
            self.input_weights = np.random.normal(0, scale= 1, size=(5, 3))
            self.chromosome = np.random.normal(0, scale= 1, size=(3, 1))
            
            ### BEST WEIGHTS FROM 3px GATE TEST ###
            '''self.input_weights = [[ 1.16207883,  1.01545432, -0.00283118],
                                [ 0.83173914,  0.19157574,  0.71581838],
                                [ 0.36122383,  0.3609113,   0.69313243],
                                [ 1.5978324,   0.80427319,  0.73569061],
                                [ 0.02832473, -0.43192903,  0.22504834]]
            self.chromosome = [[ 0.02253661],
                            [-1.49298812],
                            [ 2.15508238]]'''
            if(params["train"]):
                self.mutate(params["f_mut_odds"])

        if (self.mutant):
            self.input_weights = self.parent1.input_weights
            self.chromosome = self.parent1.chromosome
            self.mutate(params["m_mut_odds"])

        if (self.child):
            # Begin with random weights and then breed
            self.input_weights = np.random.normal(0, scale= 1, size=(5, 3))
            self.chromosome = np.random.normal(0, scale= 1, size=(3, 1))
            self.breed()
            self.mutate(params["c_mut_odds"])

    def mutate(self, MR = 0.05):
        mutation_rate = MR
        for i in range(len(self.input_weights)):
            for j in range(len(self.input_weights[i])):
                gene_code = random.randint(0, 100) * 0.01
                if (gene_code < mutation_rate):
                    learning_rate = random.randint(0, 10) * 0.005
                    even = random.randint(0, 2)
                    if(even):
                        self.input_weights[i][j] += learning_rate
                    else:
                        self.input_weights[i][j] += -1 * learning_rate

        for i in range(len(self.chromosome)):
            for j in range(len(self.chromosome[i])):
                gene_code = random.randint(0, 100) * 0.01
                if (gene_code < mutation_rate):
                    learning_rate = random.randint(0, 10) * 0.005
                    even = random.randint(0, 2)
                    if(even):
                        self.chromosome[i][j] += learning_rate
                    else:
                        self.chromosome[i][j] += -1 * learning_rate

    def breed(self):
        for i in range(len(self.input_weights)):
            for j in range(len(self.input_weights[i])):
                self.input_weights[i][j] = (self.parent1.input_weights[i][j] + self.parent2.input_weights[i][j]) / 2

        for i in range(len(self.chromosome)):
            for j in range(len(self.chromosome[i])):
                self.chromosome[i][j] = (self.parent1.chromosome[i][j] + self.parent2.chromosome[i][j]) / 2

        self.fitness = (self.parent1.fitness + self.parent2.fitness) / 2
        
    def sigmoid(self, state):
        return 1 / (1 + np.exp(-state))
        

    
