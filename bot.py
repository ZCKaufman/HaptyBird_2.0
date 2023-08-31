import pygame
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import collections
from random import randint

DEVICE = "cpu"

class HaptyBot(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        ### PARAMETERS AND VARIABLES ###
        # Training Parameters
        self.test = False
        self.learning_rate = 2.084e-05
        self.memory_size = 5000
        self.first_layer = 10
        self.second_layer = 8
        self.third_layer = 6
        # Training variables
        self.reward = 0
        self.gamma = 0.9 # Figure out what this is
        self.prediction_count = 0      
        self.epsilon = 1
        self.epsilon_decay = 0.02055
        self.memory = collections.deque(maxlen=self.memory_size)
        self.optimizer = None
        # Deployment Parameters
        self.x = 200 # Starting position
        self.y = params["cursor_y_axis"]
        self.deploy_epsilon = 0.3191
        self.x_change = 1 # Arbitrarily begin by going right
        # Training AND Deployment Parameters/Variables
        self.score = 0
        self.weights_path = "weights/weights.h5"
        self.weights = self.weights_path
        self.load_weights = False

        ### NETWORK CREATION ###
        self.f1 = nn.Linear(7, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 3)
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x

    # Training methods
    # This is NOT to be confused with score() which is part of the game process, reward() is part of training
    def get_reward(self, hit, state, params): 
        self.reward = 0
        if hit[0]:
            if(hit[1] == 1): # Went through correct gate
                self.reward = params["PGate"]
            elif(hit[1] == 0): # Went through wrong gate
                self.reward = params["NGate"]
            if(hit[1] == -1): # Hit a wall
                self.reward = params["WallHit"]
            return self.reward
        else: # CONCERN: Should I be using only ifs, multiple of these may be true so they should stack
            if state[1] and state[-2]: # Gate is right and bot moved right
                self.reward = params["PDir"]
            elif state[2] and state[-3]: # Gate is left and bot moved left
                self.reward = params["PDir"]
            elif state[1] and (state[-3] or state[-1]): # Gate is right but bot moved left or stayed still
                self.reward = params["NDir"]
            elif state[2] and (state[-2] or state[-1]): # Gate is left but bot moved right or stayed still
                self.reward = params["NDir"]
            elif state[3]: # Bot is within gate range
                self.reward = params["PRange"]
            elif state[3]: # Bot is within gate range
                self.reward = params["NRange"]
        return self.reward
    
    def move(self, params, state):
        move = [0, 0, 0]
        if random.uniform(0, 1) < self.epsilon:
                move = np.eye(3)[randint(0,2)]
        else:
            self.prediction_count += 1
            with torch.no_grad():
                prev_state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)
                prediction = self.forward(prev_state_tensor)
                move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]
            #print(move, prediction)

        if np.array_equal(move, [1, 0, 0]) or self.x < 0 or self.x > params["game_height"]: # stay
            #print("Stay")
            self.x = self.x # Stay, also activated when bot is trying to go through border
            self.x_change = 0
        elif np.array_equal(move, [0, 1, 0]):  # right
            #print("Right")
            self.x += 1
            self.x_change = 1
        elif np.array_equal(move, [0, 0, 1]):  # left
            #print("Left")
            self.x -= 1
            self.x_change = -1

        return move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return
    
    def train_LT_memory(self):
        for state, action, reward, next_state, done, in self.memory:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(next_state.reshape((1, 7)), dtype=torch.float32).to(DEVICE)
            state_tensor = torch.tensor(state.reshape((1, 7)), dtype=torch.float32, requires_grad=True).to(DEVICE)
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()

    def train_ST_memory(self, state, action, reward, next_state, done):
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(next_state.reshape((1, 7)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 7)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()

    
