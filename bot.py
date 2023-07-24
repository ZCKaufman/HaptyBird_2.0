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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class HaptyBot(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        # Training variables
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size']) #Replace with list?
        #self.memory = np.empty(params["memory_size"])
        self.optimizer = None
        # Deployment variables
        self.score = 0
        self.x = params["bot_start_x"]
        self.y = params["cursor_y_axis"]
        self.x_change = 1 # Arbitrarily begin by going right
        # Training AND Deployment variables
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']

        # Create network
        self.f1 = nn.Linear(14, self.first_layer)
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
    def get_reward(self, hit): 
        self.reward = 0
        if hit[0]:
            if(hit[1] == 1):
                self.reward = 10
            elif(hit[1] == 0):
                self.reward = -10
            elif(hit[1] == -1):
                self.reward = -30
            return self.reward
        return self.reward
    
    def move(self, params, state):
        move = [0, 0, 0]
        if random.uniform(0, 1) < self.epsilon:
                move = np.eye(3)[randint(0,2)]
        else:
            #print("Made a prediction!")
            # predict action based on the old state
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
        #np.append(self.memory, [state, action, reward, next_state, done])
        self.memory.append((state, action, reward, next_state, done))
        return
    
    def train_LT_memory(self):
        for state, action, reward, next_state, done, in self.memory:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(next_state.reshape((1, 14)), dtype=torch.float32).to(DEVICE)
            state_tensor = torch.tensor(state.reshape((1, 14)), dtype=torch.float32, requires_grad=True).to(DEVICE)
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
        next_state_tensor = torch.tensor(next_state.reshape((1, 14)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 14)), dtype=torch.float32, requires_grad=True).to(DEVICE)
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

    
