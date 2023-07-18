import pygame
import torch
import torch.optim as opt
import torch.nn as nn
import numpy as np
import pandas as pd

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
        #self.memory = collections.deque(maxlen=params['memory_size']) #Replace with list?
        self.memory = np.empty(params["memory_size"])
        self.optimizer = None
        # Deployment variables

        # Training AND Deployment variables
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']

        # Create network
        self.f1 = nn.Linear(7, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 3)
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))

    # Deployment methods
    def get_state(self, player, gate):
        state = [
            player.x not in gate.x_range,  # danger

            player.x < gate.x_range[0], # Gate right
            player.x > gate.x_range[-1], # Gate left
            player.x in gate.x_range, # In gate range

            player.x_change == -20,  # moving left
            player.x_change == 20,  # moving right
            player.x_change == 0, # Staying still
        ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)

    # Training methods
    # This is NOT to be confused with score() which is part of the game process, reward() is part of training
    def reward(self, player, crash): 
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.scored:
            self.reward = 10
        return self.reward