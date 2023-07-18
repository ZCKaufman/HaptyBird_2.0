import numpy as np
import random

class Gate:
    def __init__(self, params, game):
        self.gate_arr = np.zeros(params["game_x_axis"], int) # Creates a horizontal bar of 0s
        self.p_gate_start = random.randint()
        # Will try to generate gate to left of start pos
        if(self.p_gate_start > params["gate_size"] + params["gate_cushion"]): 
            # Changes positive gate values to 1
            self.gate_arr[self.p_gate_start - params["gate_size"]:self.p_gate_start] = 1
        
        # Will generate gate to the right of start pos
        else:
            self.gate_arr[self.p_gate_start:self.p_gate_start + params["gate_size"]] = 1

        gate_loc = np.nonzero(self.gate_arr)[0]
        # Generate negative gate
        if(gate_loc >= params["gate_cushion"] + params["gate_size"] + params["gate_min_distance"]):
            self.n_gate_start = random.randint(params["gate_cushion"] + params["gate_size"], gate_loc - params["gate_min_distance"])
            # Changes negative gate vlaues to -1
            self.gate_arr[self.n_gate_start - params["gate_size"]:self.n_gate_start] = -1
        else:
            self.n_gate_start = random.randint(gate_loc + params["gate_size"]*2 + params["gate_min_distance"], params["game_x_axis"] - params["gate_cushion"])
            self.gate_arr[self.n_gate_start:self.n_gate_start + params["gate_size"]] = -1

        print(self.gate_arr)