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
import ray
from ray import tune
from ray.tune import TuneConfig
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

DEVICE = "cpu"
CURSORS = []
WALLS = []
BOTS_TO_BREED = []
PRIZE_BOTS = []
GENERATION = 0
FITNESS = 0
RUNNING = True
#pygame.display.set_caption("HaptyBird 3.0")
#surface = None

def init_params():
    
    params = dict()
    # Bot NN parameters (adapted largely from https://github.com/maurock/snake-ga/tree/master)
    params['total_games'] = 100 #396
    # Game parameters
    params["game_width"] = 180
    params["game_height"] = 270
    params["cursor_y"] = np.floor(params["game_height"] - 5)
    params["gate_size"] = params["game_width"] / 10
    params["num_generations"] = 256
    # RAY PARAMS
    params["c_mut_odds"] = 0
    params["m_mut_odds"] = 0
    params["f_mut_odds"] = 0
    params["Generations"] = 0
    return params

def render(params):
    global CURSORS, WALLS, GENERATION
    # May have to move this to something global so it doesnt flicker
    #background = pygame.image.load("imgs/background.jpg")
    current_fit = 0
    surface = pygame.display.set_mode((params["game_width"], params["game_height"]))

    surface.fill("black")
    font = pygame.font.SysFont("Segoe UI", 12)
    font_bold = pygame.font.SysFont("Segoe UI", 12, True)
    generations = font_bold.render("GENERATION:", True, (255,255,255))
    num_generations = font.render(str(GENERATION), True, (255,255,255))
    fitness = font_bold.render("BEST FITNESS:", True, (255,255,255))
    fitness_lvl = font.render(str(FITNESS), True, (255,255,255))

    surface.blit(generations, (5,10))
    surface.blit(num_generations, (180 - 50,10))
    surface.blit(fitness, (5,20))
    surface.blit(fitness_lvl, (180 - 50,20))
        
    for i in CURSORS:
        if i.fitness > current_fit:
            current_fit = i.fitness
        color = (np.floor(255 - (i.fitness * .255)), 0, np.floor(0 + (i.fitness * .255)))
        pygame.draw.circle(surface, color, (i.x, i.y), 2) # Bot cursor

    local_fitness = font_bold.render("CURRENT FITNESS:", True, (255,255,255))
    local_fitness_lvl = font.render(str(current_fit), True, (255,255,255))

    surface.blit(local_fitness, (5,30))
    surface.blit(local_fitness_lvl, (180 - 50,30))
    
    for i in WALLS:
        pygame.draw.rect(surface, "red", (
            0, # Begin at x = 0
            i.y, # y = y
            i.left, # x = left side of gate
            3 # Arbitrary height
        ))
        pygame.draw.rect(surface, "red", (
            i.right, # x = right side of gate
            i.y, # y = y
            params["game_width"], # x = right side of game
            3 # Arbitrary height
        ))

    pygame.display.update()
    pygame.event.get()
    #if(current_fit > 50):
    #    time.sleep(params["speed"] * 0.01)
    return

def train(config): # Runs the game
    pygame.init()
    global RUNNING, GENERATION, BOTS_TO_BREED, FITNESS

    params = config["params"]
    params["c_mut_odds"] = config["c"]
    params["m_mut_odds"] = config["m"]
    params["f_mut_odds"] = config["f"]
    
    while (GENERATION < 500 and FITNESS < 750): # Training loop
        GENERATION += 1
        params["Generations"] = GENERATION

        # Check if the game has been ended by user
        '''for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()'''
                
        ### GENERATION ###
        # Prize Bots
       # for b in PRIZE_BOTS:
       #     CURSORS.append(b)

        # Breed
        for i in range(0, len(BOTS_TO_BREED) - 1):
            CURSORS.append(HaptyBaby(child=True, parent1=BOTS_TO_BREED[i], parent2=BOTS_TO_BREED[i + 1]))

        # Copy and Mutate
        for i in range(0, len(BOTS_TO_BREED)):
            CURSORS.append(BOTS_TO_BREED[i])
            CURSORS.append(HaptyBaby(mutant=True, parent1=BOTS_TO_BREED[i]))

        while(len(CURSORS) > params["num_generations"]):
            CURSORS.pop(0)

        # Mutation rate of 5, can go as high as 10, but goal is to get down to 1%
        for i in range(len(CURSORS), params["num_generations"]):
            CURSORS.append(HaptyBaby(params))

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
                if c.fitness > FITNESS:# and c.fitness > 25:
                    FITNESS = c.fitness
                    #PRIZE_BOTS.append(c)
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
    session.report({"fitness": FITNESS})

if __name__ == '__main__':
    start_time = time.time()
    print("Time started")
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = init_params()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=False)
    parser.add_argument("--speed", nargs='?', type=int, default=5)
    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    params['speed'] = args.speed

    ### RAY TUNE SETUP ###
    search_space = {
        "c": tune.uniform(0, 0.10),
        "m": tune.uniform(0, 0.10),
        "f": tune.uniform(0, 0.10),
        "params": params
    }

    ray.init(num_cpus=24, num_gpus=0, ignore_reinit_error=True)
    tuner = tune.Tuner(train, param_space=search_space, tune_config=TuneConfig(scheduler=ASHAScheduler(), metric="fitness", mode="max", num_samples=120, chdir_to_trial_dir=False))

    ### RAY RESULTS
    results = tuner.fit()
    best_results = results.get_best_result()
    print("RESULTS\n", results)
    print("\nBEST RESULT\n", best_results)

    end_time = time.time()
    print("Time ended")
    print("Total time:", end_time - start_time)