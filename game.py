import pygame
import torch
import torch.optim as opt
import numpy as np
import argparse
import distutils.util
import seaborn as sb
import matplotlib.pyplot as plt
from bot import HaptyBot
from player import Player
from gate import Gate
import datetime
import time
import optuna

DEVICE = "cpu"

# 500 Games, score -5414, decay 1/20, LR 0.01

def init_params():
    params = dict()
    # Bot NN parameters (adapted largely from https://github.com/maurock/snake-ga/tree/master)
    params['total_games'] = 500
    # Game parameters
    params["game_x_axis"] = 180
    params["game_y_axis"] = 270
    params["display_scale"] = 3
    params["game_width"] = params["game_x_axis"] * params["display_scale"]
    params["game_height"] = params["game_y_axis"] * params["display_scale"]
    params["cursor_y_axis"] = params["game_height"] - (5 * params["display_scale"])
    params["player_start_x"] = 200
    params["gate_size"] = 20
    params["gate_min_distance"] = 5
    params["gate_cushion"] = 1
    # Data parameters
    params['plot_score'] = True
    params["attempt"] = 8
    params["target_acc"] = -10
    return params

def objective(trial, params): # Runs the game
    pygame.init()
    bot = HaptyBot(params)
    bot.to(DEVICE)
    player = HaptyBot(params) # Eventually switch to Player class
    #bot.optimizer = opt.Adam(bot.parameters(), weight_decay=0, lr=bot.learning_rate)
    ### OPTUNA HYPERPARAMETERS ###
    trial.should_prune()
    bot.learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    bot.optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    bot.epsilon_decay = trial.suggest_float("decay", 1e-3, 5e-1, log=True)
    bot.deploy_epsilon = trial.suggest_float("epsilon", 0.5, 1e-2)

    # Training stuff
    gates_passed = 0
    games_counter = 0
    bot_scores = []
    game_scores = []
    acc_scores = []
    player_scores = []
    curr_acc = 0

    initial_action = [1, 0, 0] # Stay still as initial action

    while(games_counter < params["total_games"]):
        # Check if the game has been ended by user
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Create a game, a player, and a gate
        game = Game(params["game_width"], params["game_height"]) # Width, Height
        #player = Player(params) 
        gate = Gate(params, game) 
        gates_passed = 0 # Reset number of gates every game

        # Begin game and display (if applicable)
        init_state = game.get_state(player, bot, gate)
        init_move = player.move(params, init_state)
        init_reward = bot.get_reward(init_move, init_state)
        secondary_state = game.get_state(player, bot, gate)
        bot.remember(init_state, initial_action, init_reward, secondary_state, False)
        bot.train_LT_memory()

        if params["display"]:
            game.render(player, bot, gate)

        games_counter += 1 
        # Run the game until a gate has been hit
        steps = 0
        while(gates_passed % 5 != 0 or gates_passed == 0):
            # Reset crash records
            player.crash = False
            bot.crash = False

            # If training, begin with high epsilon and work up, else use param epsilon
            if bot.test == False:
                bot.epsilon = max(0, (1 - (games_counter * bot.epsilon_decay))) + bot.deploy_epsilon
            else:
                bot.epsilon = bot.deploy_epsilon

            player_hit, bot_hit = gate.update_y(params, player, bot, game)
            # Move gate downwards
            if(player_hit[0] or bot_hit[0]):
                gates_passed += 1
                gate = Gate(params, game)

            # Get state before update
            state = game.get_state(player, bot, gate)

            # Move player/bot
            player_move = player.move(params, state)
            bot_move = bot.move(params, state) # This will do all the decision making and position reassignment for the bot

            new_state = game.get_state(player, bot, gate)
            # Set scores and/or rewards as relevant
            if(bot.test == False):
                reward = bot.get_reward(bot_hit, new_state)
                bot.train_ST_memory(state, bot_move, reward, new_state, bot_hit[0])
                bot.remember(state, bot_move, reward, new_state, bot_hit[0])
            if(player_hit[0] or bot_hit[0]):
                player_scores.append(score(player, player_hit))
                bot_scores.append(score(bot, bot_hit))
                #break

            if params['display']:
                game.render(player, bot, gate)
                pygame.time.wait(params["speed"]) # Slows down game for viewing

            steps += 1
        curr_acc = bot.score / games_counter
        acc_scores.append(curr_acc)
        game_scores.append(sum(bot_scores[-5:]))
        print(f'Game {games_counter}\tBot: {sum(bot_scores[-5:])}, {bot.score}\tPlayer: {sum(player_scores[-5:])}, {player.score}\tEpsilon: {round(bot.epsilon, 2)}\tAccurracy: {round(curr_acc, 4)}')
    plot_scores(game_scores, bot_scores, acc_scores, params["attempt"])
    if bot.test == False and curr_acc > params["target_acc"]:
        model_weights = bot.state_dict()
        torch.save(model_weights, bot.weights_path)
        print(f'Weights saved, finished with higher accuracy\nCurrent: {curr_acc}\nPrevious Best: {params["target_acc"]}')
    elif bot.test:
        print("Finished testing.")
    else:
        print(f'Finished with lower accuracy\nCurrent: {curr_acc}\nTarget: {params["target_acc"]}')
    return curr_acc

def score(user, gate_interact):
    if(gate_interact[1] == 1):
        user.score += 1
        return 1
    elif(gate_interact[1] == 0):
        user.score -= 1
        return -1
    elif(gate_interact[1] == -1):
        user.score -= 3
        return -3
    
def plot_scores(game_scores, bot_scores, acc_scores, attempt):
    x = len(bot_scores)

    plt.plot(game_scores, "bo")
    plt.plot(acc_scores)
    plt.savefig("plots/plot" + str(attempt) + ".png")

class Game:
    def __init__(self, width, height):
        pygame.display.set_caption("HaptyBird 2.0")
        self.game_width = width
        self.game_height = height
        self.surface = pygame.display.set_mode((width, height))
        self.background = pygame.image.load("imgs/background.jpg")
        
    def render(self, player, bot, gate):
        # UI display
        #self.surface = pygame.display.set_mode((self.game_width, self.game_height)) # There has GOT to be a better way
        self.surface.fill("black")
        font = pygame.font.SysFont("Segoe UI", 20)
        font_bold = pygame.font.SysFont("Segoe UI", 20, True)
        player_score = font_bold.render("PLAYER 1:", True, (255,255,255))
        player_score_n = font.render(str(player.score), True, (255,255,255))
        bot_score = font_bold.render("PLAYER 2:", True, (255,255,255))
        bot_score_n = font.render(str(bot.score), True, (255,255,255))

        self.surface.blit(player_score, (self.game_width / 5,10))
        self.surface.blit(player_score_n, (2*self.game_width / 5,10))
        self.surface.blit(bot_score, (3*self.game_width / 5,10))
        self.surface.blit(bot_score_n, (4*self.game_width / 5,10))
        self.surface.blit(player_score, (5*self.game_width / 5,10))
        # Game display
        pygame.draw.circle(self.surface, "blue", (player.x, player.y), 5) # Player cursor
        pygame.draw.circle(self.surface, "red", (bot.x, bot.y), 5) # Bot cursor
        gate.display(params, self) # Gate is complex, it has to draw itself

        pygame.display.update()
        pygame.event.get()

    def get_state(self, player, bot, gate):
        # This state is used to train the bot, if modified it will break things
        pos_gate = gate.pos_gate[0]
        '''state = [
            # Player state stats
            player.x not in pos_gate, # Is the player in danger
            player.x < pos_gate[0], # Gate right
            player.x > pos_gate[-1], # Gate left
            player.x in pos_gate, # In gate range
            player.x_change == -1,  # move left
            player.x_change == 1,  # move right
            player.x_change == 0, # Stay still

            # Bot state stats
            bot.x not in pos_gate, # Is the player in danger
            bot.x < pos_gate[0], # Gate right
            bot.x > pos_gate[-1], # Gate left
            bot.x in pos_gate, # In gate range
            bot.x_change == -1,  # move left
            bot.x_change == 1,  # move right
            bot.x_change == 0 # Stay still
        ]'''

        state = [
            # Bot state stats
            bot.x not in pos_gate, # Is the player in danger
            bot.x < pos_gate[0], # Gate right
            bot.x > pos_gate[-1], # Gate left
            bot.x in pos_gate, # In gate range
            bot.x_change == -1,  # move left
            bot.x_change == 1,  # move right
            bot.x_change == 0 # Stay still
        ]

        # Convert state to 1s and 0s to help with training and make things easier
        state = [int(i) for i in state]
        return np.asarray(state)

if __name__ == '__main__':
    start_time = time.time()
    print("Time started")
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = init_params()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=False)
    parser.add_argument("--speed", nargs='?', type=int, default=0)
    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    params['speed'] = args.speed

    ####################
    ### OPTUNA SETUP ###
    ####################
    study = optuna.create_study(direction="maximize")  # 'maximize' because objective function is returning accuracy
    #study = optuna.create_study(direction="minimize")  # 'minimize' because objective function is returning loss
    study.optimize(lambda trial: objective(trial, params), n_trials=500)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    end_time = time.time()
    print("Time ended")
    print("Total time:", end_time - start_time)