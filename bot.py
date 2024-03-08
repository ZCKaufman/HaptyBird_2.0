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
    def __init__(self, params, first_gen = True, mutant = False, child = False, parent1 = None, parent2 = None, gate_choice = 0):
        ### INPUT: Params
        ### OUTPUT: None
        ### DESCRIPTION: Generates a baby bot. Could be a first generation, mutant, or child

        # GENETIC VARIABLES #
        self.alive = True
        self.fitness = 0
        self.best = False 
        self.chromosome_1 = []
        self.chromosome_2 = []
        self.chromosome_3 = []
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
        self.gate_choice = gate_choice

        # GAME STATISTICS #
        self.score = 0

        self.set_weights(params)

    def update_state(self, wall):
        if(self.gate_choice == wall.left_gate[2]):
            self.dist_c_gate_l = wall.left_gate[0]
            self.dist_c_gate_r = wall.left_gate[1]
            self.dist_o_gate_l = wall.right_gate[0]
            self.dist_o_gate_r = wall.right_gate[1]
        else:
            self.dist_c_gate_l = wall.right_gate[0]
            self.dist_c_gate_r = wall.right_gate[1]
            self.dist_o_gate_l = wall.left_gate[0]
            self.dist_o_gate_r = wall.left_gate[1]
        self.dist_y = self.y - wall.y
        if(self.gate_choice == wall.left_gate[2]):
            if(self.x > wall.left_gate[0] and self.x < wall.left_gate[1]):
                self.fitness += 0.01 # Reward for being in the right spot
        else:
            if(self.x > wall.right_gate[0] and self.x < wall.right_gate[1]):
                self.fitness += 0.01 # Reward for being in the right spot
        if(self.fitness > 150):
            print("-----------", self.fitness, "----------")
            print("1:", self.chromosome_1)
            print("2:", self.chromosome_2)
            print("3:", self.chromosome_3)
            print("-----------", self.fitness, "----------")

    def collision(self, wall):
        if(self.gate_choice == wall.left_gate[2]):
            if(self.x > wall.left_gate[0] and self.x < wall.left_gate[1]):
                self.passed += 1
            elif(self.x > wall.right_gate[0] and self.x < wall.right_gate[1]):
                self.passed += .5
            else:
                self.score -= 1
                self.fitness -= 3
                self.alive = False

    def move(self, params):
        # Set the inputs to state
        state = [self.x, self.dist_c_gate_l, self.dist_c_gate_r, self.dist_o_gate_l, self.dist_o_gate_r, self.dist_y, self.prev_move]

        layer1 = np.dot(state, self.chromosome_1)
        layer1_a = self.sigmoid(layer1)
        layer2 = np.dot(layer1_a, self.chromosome_2)
        layer2_a = self.tanh(layer2)
        layer3 = np.dot(layer2_a, self.chromosome_3)
        prediction = self.tanh(layer3)

        if (prediction < -0.33 and self.x > 0):
            self.x -= 1
            self.prev_move = -1
        elif (prediction > 0.33 and self.x < params["game_width"]):
            self.x += 1
            self.prev_move = 1
        else:
            self.x = self.x
            self.prev_move = 0

    def set_weights(self, params):
        if (self.first_gen):
            rand_chance = random.random()
            if(rand_chance < 0.0):
                ### RANDOM WEIGHTS ###
                self.chromosome_1 = np.random.normal(0, scale= 1, size=(7, 4))
                self.chromosome_2 = np.random.normal(0, scale= 1, size=(4, 3))
                self.chromosome_3 = np.random.normal(0, scale= 1, size=(3, 1))
            ### TRAINING FITNESS: 210 ###
            else:
                self.chromosome_1 = [[ 1.73530077,  2.47868579,  1.20606162,  1.857809],
                                    [-0.55773877,  0.89342955, -1.00407992, -0.03916429],
                                    [-0.58612374,  0.28941589, -1.06137233,  0.17453236],
                                    [-1.4362588,  -0.67199804,  0.41738395, -0.83101048],
                                    [ 0.70089289,  0.37938383,  0.74253104, -1.15228495],
                                    [ 0.06057514, -1.65373908,  1.18812516,  2.05190788],
                                    [ 1.23858558,  0.61391846,  0.22784838,  0.68628033]]
                self.chromosome_2 = [[ 0.16616632, -0.12397955, -2.24033337],
                                    [-0.69633946,  2.2364728,   0.3751559 ],
                                    [ 1.06409332,  0.54712497,  0.50106843],
                                    [-0.72125204, -0.26966407,  0.16936832]]
                self.chromosome_3 = [[1.85900397],
                                    [1.07706974],
                                    [1.32492735]]
            if(params["train"]):
                self.mutate(params["f_mut_odds"])

        if (self.mutant):
            self.chromosome_1 = self.parent1.chromosome_1
            self.chromosome_2 = self.parent1.chromosome_2
            self.chromosome_3 = self.parent1.chromosome_3
            self.mutate(params["m_mut_odds"])

        if (self.child):
            # Begin with random weights and then breed
            self.chromosome_1 = np.random.normal(0, scale= 1, size=(7, 4))
            self.chromosome_2 = np.random.normal(0, scale= 1, size=(4, 3))
            self.chromosome_3 = np.random.normal(0, scale= 1, size=(3, 1))
            self.breed()
            self.mutate(params["c_mut_odds"])

    def mutate(self, MR = 0.05):
        mutation_rate = MR
        for i in range(len(self.chromosome_1)):
            for j in range(len(self.chromosome_1[i])):
                gene_code = random.randint(0, 100) * 0.01
                if (gene_code < mutation_rate):
                    learning_rate = random.randint(0, 10) * 0.005
                    even = random.randint(0, 2)
                    if(even):
                        self.chromosome_1[i][j] += learning_rate
                    else:
                        self.chromosome_1[i][j] += -1 * learning_rate

        for i in range(len(self.chromosome_2)):
            for j in range(len(self.chromosome_2[i])):
                gene_code = random.randint(0, 100) * 0.01
                if (gene_code < mutation_rate):
                    learning_rate = random.randint(0, 10) * 0.005
                    even = random.randint(0, 2)
                    if(even):
                        self.chromosome_2[i][j] += learning_rate
                    else:
                        self.chromosome_2[i][j] += -1 * learning_rate
        
        for i in range(len(self.chromosome_3)):
            for j in range(len(self.chromosome_3[i])):
                gene_code = random.randint(0, 100) * 0.01
                if (gene_code < mutation_rate):
                    learning_rate = random.randint(0, 10) * 0.005
                    even = random.randint(0, 2)
                    if(even):
                        self.chromosome_3[i][j] += learning_rate
                    else:
                        self.chromosome_3[i][j] += -1 * learning_rate

    def breed(self):
        for i in range(len(self.chromosome_1)):
            for j in range(len(self.chromosome_1[i])):
                self.chromosome_1[i][j] = (self.parent1.chromosome_1[i][j] + self.parent2.chromosome_1[i][j]) / 2

        for i in range(len(self.chromosome_2)):
            for j in range(len(self.chromosome_2[i])):
                self.chromosome_2[i][j] = (self.parent1.chromosome_2[i][j] + self.parent2.chromosome_2[i][j]) / 2

        for i in range(len(self.chromosome_3)):
            for j in range(len(self.chromosome_3[i])):
                self.chromosome_3[i][j] = (self.parent1.chromosome_3[i][j] + self.parent2.chromosome_3[i][j]) / 2

        self.fitness = 0#(self.parent1.fitness + self.parent2.fitness) / 2
        
    def sigmoid(self, state):
        return 1 / (1 + np.exp(-state))
    
    def tanh(self, state):
        return (2 / (1 + np.exp(2 * -state))) - 1
    
    def relu(self, state):
        return np.max(0, state)
        

    
