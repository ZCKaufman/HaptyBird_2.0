import pygame
import torch
import torch.optim as opt
import numpy as np
import argparse
import distutils.util
import seaborn as sb
import matplotlib.pyplot as plt
from bot import HaptyBaby
from player import Player
from wall import Wall
import datetime
import time

CURSORS = []
WALLS = []
BOTS_TO_BREED = []
GENERATION = 0
FITNESS = 0
RUNNING = True
surface = None

def init_params():
    global surface

    params = dict()
    # Game parameters
    params["game_width"] = 180
    params["game_height"] = 270
    params["cursor_y"] = np.floor(params["game_height"] - 5)
    params["gate_size"] = 3
    params["num_generations"] = 128
    params["c_mut_odds"] = 0.0274#0.011905250442820692#0.06718772399427941
    params["m_mut_odds"] = 0.0025#0.008419509346686838#0.047894877742536146
    params["f_mut_odds"] = 0.0813#0.05969429389680664#0.09902455284411923
    params["train"] = True
    return params

def render(params):
    global CURSORS, WALLS, GENERATION
    current_fit = 0

    surface.fill("black")
    font = pygame.font.SysFont("Segoe UI", 12)
    font_bold = pygame.font.SysFont("Segoe UI", 12, True)
    generations = font_bold.render("GENERATION:", True, (255,255,255))
    num_generations = font.render(str(GENERATION), True, (255,255,255))
    max_fit_txt = font_bold.render("MAX FITNESS:", True, (255,255,255))
    max_fit = font.render(str(FITNESS), True, (255,255,255))
    #score = font_bold.render("SCORE:", True, (255,255,255))
    #bot_score = font.render(str(CURSORS[-1].score), True, (255,255,255))

    surface.blit(generations, (5,10))
    surface.blit(num_generations, (180 - 50,10))
    if(params["train"]):
        surface.blit(max_fit_txt, (5,20))
        surface.blit(max_fit, (180 - 50,20))
    #else:
    #    surface.blit(score, (5,20))
    #    surface.blit(bot_score, (180 - 50,20))
        
    for i in CURSORS:
        if i.fitness > current_fit:
            current_fit = i.fitness
        color = "red"
        pygame.draw.circle(surface, color, (i.x, i.y), 2) # Bot cursor

    local_fitness = font_bold.render("CURRENT FITNESS:", True, (255,255,255))
    local_fitness_lvl = font.render(str(current_fit), True, (255,255,255))

    surface.blit(local_fitness, (5,30))
    surface.blit(local_fitness_lvl, (180 - 50,30))
    
    for i in WALLS:
        # Left wall
        pygame.draw.rect(surface, "red", (
            0, # Begin at x = 0
            i.y, # y = y
            i.left_gate[0], # x = left side of left gate
            3 # Arbitrary height
        ))
        # Left gate
        if(i.left_gate[2]):
            left_color = "green"
            right_color = "blue"
        else:
            left_color = "blue"
            right_color = "green"
        pygame.draw.rect(surface, left_color, (
            i.left_gate[0], # Begin at x = 0
            i.y, # y = y
            i.left_gate[1], # x = left side of left gate
            3 # Arbitrary height
        ))
        # Center wall
        pygame.draw.rect(surface, "red", (
            i.left_gate[1], # x = right side of gate
            i.y, # y = y
            i.right_gate[0], # x = right side of game
            3 # Arbitrary height
        ))
        # Right gate
        pygame.draw.rect(surface, right_color, (
            i.right_gate[0], # Begin at x = 0
            i.y, # y = y
            i.right_gate[1], # x = left side of left gate
            3 # Arbitrary height
        ))
        # Right wall
        pygame.draw.rect(surface, "red", (
            i.right_gate[1], # x = right side of gate
            i.y, # y = y
            params["game_width"], # x = right side of game
            3 # Arbitrary height
        ))

    pygame.display.update()
    pygame.event.get()

    time.sleep(params["speed"] * 0.01)
    
    return

def train(params): # Runs the game
    pygame.init()
    global RUNNING, GENERATION, BOTS_TO_BREED, FITNESS
    
    while (FITNESS < 1750): # Training loop
        GENERATION += 1
        if(GENERATION % 50 == 0):
            print(GENERATION, FITNESS)
        # Check if the game has been ended by user
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Breed
        for i in range(0, len(BOTS_TO_BREED) - 1):
            CURSORS.append(HaptyBaby(params, child=True, parent1=BOTS_TO_BREED[i], parent2=BOTS_TO_BREED[i + 1]))

        # Copy and Mutate
        for i in range(0, len(BOTS_TO_BREED)):
            CURSORS.append(BOTS_TO_BREED[i])
            CURSORS.append(HaptyBaby(params, mutant=True, parent1=BOTS_TO_BREED[i]))

        while(len(CURSORS) > params["num_generations"]):
            CURSORS.pop(0)

        # Mutation rate of 5, can go as high as 10, but goal is to get down to 1%
        for i in range(len(CURSORS), params["num_generations"]):
            CURSORS.append(HaptyBaby(params))

        RUNNING = True

        ### GAME ###
        while(RUNNING): # Game loop
            bots_to_breed = []

            if(not len(WALLS)):
                WALLS.append(Wall(params))
            else:
                if(WALLS[-1].y >= 160):
                    odds = np.random.randint(0, 100)
                    if(odds <= 50):
                        WALLS.append(Wall(params))

            for c in CURSORS:
                if c.fitness > FITNESS:# and c.fitness > 25:
                    FITNESS = c.fitness
                    #PRIZE_BOTS.append(c)

                    # print("FITNESS:", c.fitness)
                    # print("CHROMOSOME 1:", c.chromosome_1)
                    # print("CHROMOSOME 2:", c.chromosome_2)
                    # print("CHROMOSOME 3:", c.chromosome_3)
                c.update_state(WALLS[0])
                c.move(params)
            
            for w in WALLS:
                if (w.update_y(params)):
                    RUNNING = False
                    bots_to_breed = []
                    for i, c in enumerate(CURSORS):
                        c.collision(w)
                        if (c.alive):
                            RUNNING = True
                            bots_to_breed.append(c)
                        else:
                            for j in bots_to_breed:
                                while(len(BOTS_TO_BREED) >= 48):
                                    BOTS_TO_BREED.pop(0)
                                BOTS_TO_BREED.append(j)
                            CURSORS.pop(i)

            if(WALLS[0].y > params["cursor_y"]):
                WALLS.pop(0)

            if (params["display"]):
                render(params)

def test(params):
    pygame.init()
    global RUNNING, GENERATION, BOTS_TO_BREED, FITNESS

    while (FITNESS < 750): # Training loop
        GENERATION += 1

        CURSORS.append(HaptyBaby(params, gate_choice=0))
        CURSORS.append(HaptyBaby(params, gate_choice=1))

        RUNNING = True

        ### GAME ###
        while(RUNNING): # Game loop

            if(not len(WALLS)):
                WALLS.append(Wall(params))
            else:
                if(WALLS[-1].y >= 160):
                    odds = np.random.randint(0, 100)
                    if(odds <= 50):
                        WALLS.append(Wall(params))

            for c in CURSORS:
                c.update_state(WALLS[0])
                c.move(params)
            
            for w in WALLS:
                if (w.update_y(params)):
                    RUNNING = False
                    BOTS_TO_BREED = []
                    for i, c in enumerate(CURSORS):
                        c.collision(w)
                        if (c.alive):
                            RUNNING = True
                            BOTS_TO_BREED.append(c)
                        else:
                            CURSORS.pop(i)

            if(WALLS[0].y > params["cursor_y"]):
                WALLS.pop(0)

            if (params["display"]):
                render(params)

if __name__ == '__main__':
    start_time = time.time()
    print("Time started")
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = init_params()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=1)
    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    params['speed'] = args.speed

    if(params["display"]):
        pygame.display.set_caption("HaptyBird 3.0")
        surface = pygame.display.set_mode((params["game_width"], params["game_height"]))

    if(params["train"]):
        train(params)
    else:
        test(params)

    end_time = time.time()
    print("Time ended")
    print("Total time:", end_time - start_time)