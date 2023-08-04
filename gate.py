import numpy as np
import random
import sys
import pygame
np.set_printoptions(threshold=sys.maxsize)

class Gate:
    def __init__(self, params, game):
        self.y = 0
        self.gate_arr = np.zeros(params["game_width"], int) # Creates a horizontal bar of 0s
        self.neg_gate = np.where(self.gate_arr < 0)
        self.pos_gate = np.where(self.gate_arr > 0)

        self.generate_gate(params)

    def generate_gate(self, params):
        self.p_gate_start = random.randint(0,params["game_width"])
        
        # Will try to generate gate to left of start pos
        if(self.p_gate_start < params["game_width"] - ((params["gate_size"] + params["gate_cushion"]) * params["display_scale"])): 
            # Changes positive gate values to 1
            self.gate_arr[self.p_gate_start:self.p_gate_start + (params["gate_size"] * params["display_scale"])] = 1
        # Will generate gate to the right of start pos
        else:
            self.gate_arr[self.p_gate_start - (params["gate_size"] * params["display_scale"]):self.p_gate_start] = 1
        gate_loc = np.where(self.gate_arr != 0)
        gate_loc = gate_loc[0]
        # Generate negative gate
        left_right = random.randint(1, 10) # Determines if the neg gate will attempt to generate left or right of the pos gate
        if(left_right < 5): # LEFT
            if(gate_loc[0] >= (params["gate_cushion"] + params["gate_size"] + params["gate_min_distance"]) * params["display_scale"]): # Expand right
                self.n_gate_start = random.randint(((params["gate_cushion"] + params["gate_size"]) * params["display_scale"]), gate_loc[0] - (params["gate_min_distance"] * params["display_scale"]))
                # Changes negative gate vlaues to -1
                self.gate_arr[self.n_gate_start - (params["gate_size"] * params["display_scale"]):self.n_gate_start] = -1
            else:
                self.n_gate_start = random.randint(gate_loc[-1] + (params["gate_min_distance"] * params["display_scale"]), params["game_width"] - (params["gate_cushion"] * params["display_scale"]))
                self.gate_arr[self.n_gate_start:self.n_gate_start + (params["gate_size"] * params["display_scale"])] = -1
        else: # RIGHT
            if(gate_loc[-1] <= (params["game_width"] - (params["gate_cushion"] + params["gate_size"] + params["gate_min_distance"]) * params["display_scale"])): # Expand right
                self.n_gate_start = random.randint((gate_loc[-1] + (params["gate_min_distance"] * params["display_scale"])), params["game_width"] - (params["gate_cushion"] * params["display_scale"]))
                # Changes negative gate vlaues to -1
                self.gate_arr[self.n_gate_start: self.n_gate_start + (params["gate_size"] * params["display_scale"])] = -1
            else:
                self.n_gate_start = random.randint(((params["gate_cushion"] + params["gate_size"]) * params["display_scale"]), gate_loc[0] - (params["gate_min_distance"] * params["display_scale"]))
                self.gate_arr[self.n_gate_start - (params["gate_size"] * params["display_scale"]):self.n_gate_start] = -1
        self.neg_gate = np.where(self.gate_arr < 0)
        self.pos_gate = np.where(self.gate_arr > 0)

    def display(self, params, game):
        self.neg_gate = np.where(self.gate_arr < 0)
        self.pos_gate = np.where(self.gate_arr > 0)

        if(self.neg_gate[0][0] < self.pos_gate[0][0]):
            # Left wall
            pygame.draw.rect(game.surface, (255,255,0), (
                0, # Start (x) on the far left
                self.y * params["display_scale"], # Start (y) wherever the gate is supposed to be
                self.neg_gate[0][0], # Have a width that spans from 0 to the beginning of the -gate
                5 * params["display_scale"], # Arbitrary height
            ))
            # Neg score
            font_bold = pygame.font.SysFont("Segoe UI", 5 * params["display_scale"], True)
            neg_gate_score = font_bold.render("$1.00", True, (255,0,0))
            game.surface.blit(neg_gate_score, (np.mean(self.neg_gate) - (5 * params["display_scale"]),self.y * params["display_scale"]))
            # Middle wall
            pygame.draw.rect(game.surface, (255,255,0), (
                self.neg_gate[0][-1], # Start (x) on the far left
                self.y * params["display_scale"], # Start (y) wherever the gate is supposed to be
                self.pos_gate[0][0] - self.neg_gate[0][-1], # Have a width that spans from 0 to the beginning of the +gate
                5 * params["display_scale"], # Arbitrary height
            ))
            # Pos score
            font_bold = pygame.font.SysFont("Segoe UI", 5 * params["display_scale"], True)
            pos_gate_score = font_bold.render("$1.23", True, (0,255,0))
            game.surface.blit(pos_gate_score, (np.mean(self.pos_gate) - (5 * params["display_scale"]),self.y * params["display_scale"]))
            # Right wall
            pygame.draw.rect(game.surface, (255,255,0), (
                self.pos_gate[0][-1], # Start (x) on the far left
                self.y * params["display_scale"], # Start (y) wherever the gate is supposed to be
                params["game_width"] - self.pos_gate[0][-1], # Have a width that spans from 0 to the beginning of the -gate
                5 * params["display_scale"], # Arbitrary height
            ))
        else:
            # Left wall
            pygame.draw.rect(game.surface, (255,255,0), (
                0, # Start (x) on the far left
                self.y * params["display_scale"], # Start (y) wherever the gate is supposed to be
                self.pos_gate[0][0], # Have a width that spans from 0 to the beginning of the -gate
                5 * params["display_scale"], # Arbitrary height
            ))
            # Pos score
            font_bold = pygame.font.SysFont("Segoe UI", 5 * params["display_scale"], True)
            pos_gate_score = font_bold.render("$1.23", True, (0,255,0))
            game.surface.blit(pos_gate_score, (np.mean(self.pos_gate) - (5 * params["display_scale"]),self.y * params["display_scale"]))
            # Middle wall
            pygame.draw.rect(game.surface, (255,255,0), (
                self.pos_gate[0][-1], # Start (x) on the far left
                self.y * params["display_scale"], # Start (y) wherever the gate is supposed to be
                self.neg_gate[0][0] - self.pos_gate[0][-1], # Have a width that spans from 0 to the beginning of the +gate
                5 * params["display_scale"], # Arbitrary height
            ))
            # Neg score
            font_bold = pygame.font.SysFont("Segoe UI", 5 * params["display_scale"], True)
            neg_gate_score = font_bold.render("$1.00", True, (255,0,0))
            game.surface.blit(neg_gate_score, (np.mean(self.neg_gate) - (5 * params["display_scale"]),self.y * params["display_scale"]))
            # Right wall
            pygame.draw.rect(game.surface, (255,255,0), (
                self.neg_gate[0][-1], # Start (x) on the far left
                self.y * params["display_scale"], # Start (y) wherever the gate is supposed to be
                params["game_width"] - self.neg_gate[0][-1], # Have a width that spans from 0 to the beginning of the -gate
                5 * params["display_scale"], # Arbitrary height
            ))
            

    def update_y(self, params, player, bot, game):
        player_output = [False, -2] # Returns a True if gate hit player, a 1 if pos gate, 0 if neg gate, -1 if wall
        bot_output = [False, -2]
        if self.y * params["display_scale"] >= player.y:
            player_output[0] = True
            if(player.x in self.pos_gate[0]):
                player_output[1] = 1
            elif(player.x in self.neg_gate[0]):
                player_output[1] = 0
            else:
                player_output[1] = -1

            bot_output[0] = True
            if(bot.x in self.pos_gate[0]):
                bot_output[1] = 1
            elif(bot.x in self.neg_gate[0]):
                bot_output[1] = 0
            else:
                bot_output[1] = -1
            #print("Hit a gate", player_output, bot_output)
            return player_output, bot_output
        else:
            #print("Ys:", self.y, self.y * params["display_scale"], player.y)
            self.y += 1
            self.display(params, game)
            return player_output, bot_output