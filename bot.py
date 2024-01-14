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
        self.gate_choice = 1

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
        self.fitness += 0.01

        if(self.fitness >= 500):
            print("FITNESS:", self.fitness)
            print("CHROMOSOME 1:", self.chromosome_1)
            print("CHROMOSOME 2:", self.chromosome_2)
            print("CHROMOSOME 3:", self.chromosome_3)

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
        layer1_a = self.tanh(layer1)
        layer2 = np.dot(layer1_a, self.chromosome_2)
        layer2_a = self.tanh(layer2)
        layer3 = np.dot(layer2_a, self.chromosome_3)
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
            self.chromosome_1 = np.random.normal(0, scale= 1, size=(7, 5))
            self.chromosome_2 = np.random.normal(0, scale= 1, size=(5, 3))
            self.chromosome_3 = np.random.normal(0, scale= 1, size=(3, 1))
            
            ### BEST WEIGHTS FROM 3px GATE TEST #
            '''self.chromosome_1 = [[ 0.46093351,  0.76599756,  0.51725556, -1.27992041, -1.73067638],
                                [-0.47011362, -0.68811518, -0.13676006, -0.06401243,  0.67630845],
                                [-0.29447367, -0.09740023,  0.36656525,  1.09661885,  0.23134315],
                                [-1.46634215, -1.09772655,  1.18300407,  0.50472777,  1.11807935],
                                [-0.82736888,  1.16051323,  0.65782399,  0.16305401, -0.88351831],
                                [-0.72433429, -0.96959554, -0.03757411,  0.15349503,  0.13077109],
                                [-0.03411067,  1.3594317,  -0.37795948,  0.49874281,  1.79486994]]
            self.chromosome_2 = [[ 0.06682074,  2.05700841,  0.39529383],
                                [ 2.69949168,  0.6734257,   1.95217545],
                                [ 0.51488829, -0.84409397, -0.27587867],
                                [ 1.32181397, -1.25127511,  1.32413136],
                                [ 2.29900897, -0.43436434, -1.83066862]]
            self.chromosome_3 = [[ 1.47157547],
            [ 1.2740228 ],
            [-2.60040346]]'''
            if(params["train"]):
                self.mutate(params["f_mut_odds"])

        if (self.mutant):
            self.chromosome_1 = self.parent1.chromosome_1
            self.chromosome_2 = self.parent1.chromosome_2
            self.chromosome_3 = self.parent1.chromosome_3
            self.mutate(params["m_mut_odds"])

        if (self.child):
            # Begin with random weights and then breed
            self.chromosome_1 = np.random.normal(0, scale= 1, size=(7, 5))
            self.chromosome_2 = np.random.normal(0, scale= 1, size=(5, 3))
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

        self.fitness = (self.parent1.fitness + self.parent2.fitness) / 2
        
    def sigmoid(self, state):
        return 1 / (1 + np.exp(-state))
    
    def tanh(self, state):
        return (2 / (1 + np.exp(2 * -state))) - 1
    
    def relu(self, state):
        return np.max(0, state)
        

    
