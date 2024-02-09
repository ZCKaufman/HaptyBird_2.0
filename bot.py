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
    def __init__(self, params, first_gen = True, mutant = False, child = False, parent1 = None, parent2 = None, gate_choice = 1):
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
        '''if(self.gate_choice == wall.left_gate[2]):
            self.dist_c_gate = wall.left_center - self.x
            self.dist_o_gate = wall.right_center - self.x
        else:
            self.dist_c_gate = wall.right_center - self.x
            self.dist_o_gate = wall.left_center - self.x'''
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
            elif(self.x > wall.right_gate[0] and self.x < wall.right_gate[1]):
                self.fitness = self.fitness # No change in fitness for being within the wrong gate range
            else:
                self.fitness -= 0.01 # Neg reward for being in the wrong spot
        else:
            if(self.x > wall.right_gate[0] and self.x < wall.right_gate[1]):
                self.fitness += 0.01 # Reward for being in the right spot
            elif(self.x > wall.left_gate[0] and self.x < wall.left_gate[1]):
                self.fitness = self.fitness # No change in fitness for being within the wrong gate range
            else:
                self.fitness -= 0.01 # Neg reward for being in the wrong spot
        if(self.fitness > 500):
            print("1:", str(self.chromosome_1))
            print("2:", str(self.chromosome_2))
            print("3:", str(self.chromosome_3))

    def collision(self, wall):
        if(self.gate_choice == wall.left_gate[2]):
            if(self.x > wall.left_gate[0] and self.x < wall.left_gate[1]):
                self.score += 1
                self.fitness += 3
                self.passed += 1
            elif(self.x > wall.right_gate[0] and self.x < wall.right_gate[1]):
                self.score += .5
                self.fitness += 1
                self.passed += .5
            else:
                self.score -= 1
                self.fitness -= 3
                self.alive = False

    def move(self, params):
        # Set the inputs to state
        #state = [self.x, self.dist_left, self.dist_right, self.dist_y, self.prev_move]
        state = [self.x, self.dist_c_gate_l, self.dist_c_gate_r, self.dist_o_gate_l, self.dist_o_gate_r, self.dist_y, self.prev_move]

        layer1 = np.dot(state, self.chromosome_1)
        layer1_a = self.sigmoid(layer1)
        layer2 = np.dot(layer1_a, self.chromosome_2)
        layer2_a = self.tanh(layer2)
        layer3 = np.dot(layer2_a, self.chromosome_3)
        prediction = self.tanh(layer3)

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
            self.chromosome_1 = np.random.normal(0, scale= 1, size=(7, 4))
            self.chromosome_2 = np.random.normal(0, scale= 1, size=(4, 3))
            self.chromosome_3 = np.random.normal(0, scale= 1, size=(3, 1))
            ### TRAINING FITNESS: 1750 ###
            
            # self.chromosome_1 = [[-0.83255689, -0.24677735,  0.7521962,   0.64078756],
            #                     [ 0.42143726,  0.25700747,  0.47326644,  0.22223561],
            #                     [ 0.41761297, -1.59884992,  0.34140426, -1.14388084],
            #                     [ 1.18319561,-1.33439961, -0.52975154,  1.46716971],
            #                     [-1.1744866,   0.34043304, -0.07638435, -0.44695094],
            #                     [ 0.84053683, -1.09521061,  0.09475642,  0.52785842],
            #                     [ 1.61743526,  0.8425844,   0.22686693,  0.6618548 ]]
            # self.chromosome_2 = [[ 0.49058185, -0.7131381,   2.35006918],
            #                     [ 0.49573697, 0.11144374,  0.41243955],
            #                     [ 0.43183828, -0.91827516,  0.51124984],
            #                     [-1.18493213,  1.07524469, -0.96786005]]
            # self.chromosome_3 = [[ 0.09001661],
            #                     [-0.44033626],
            #                     [ 1.71977779]]
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
        

    
