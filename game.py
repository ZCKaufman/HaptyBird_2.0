import pygame
import torch
import torch.optim as opt
import numpy as np
import argparse
import distutils.util
from bot import HaptyBot
from player import Player
from gate import Gate
import datetime

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_params():
    params = dict()
    # Bot NN parameters (adapted largely from https://github.com/maurock/snake-ga/tree/master)
    params['epsilon_decay'] = 1/100
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 1    # Neurons in the first layer
    params['second_layer_size'] = 20   # Neurons in the second layer
    params['third_layer_size'] = 50    # Neurons in the third layer
    params['total_gates'] = 180        
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    params['train'] = True # Do not change, use arguments in terminal to train
    params["deploy"] = False # Do not change, HaptyBot deployed to game should be default
    # Game parameters
    params["game_x_axis"] = 180
    params["game_y_axis"] = 270
    params["display_scale"] = 3
    params["game_width"] = params["game_x_axis"] * params["display_scale"]
    params["game_height"] = params["game_y_axis"] * params["display_scale"]
    params["deploy_epsilon"] = 0.01
    params["cursor_y_axis"] = params["game_height"] - (5 * params["display_scale"])
    params["bot_start_x"] = 100
    params["player_start_x"] = 200
    params["gate_size"] = 20
    params["gate_min_distance"] = 5
    params["gate_cushion"] = 1
    # Data parameters
    params['weights_path'] = 'weights/weights.h5'
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'

    # User parameters

    return params

def run_games(params): # Runs the game
    pygame.init()
    bot = HaptyBot(params)
    player = HaptyBot(params) 
    bot.optimizer = opt.Adam(bot.parameters(), weight_decay=0, lr=params['learning_rate'])
    # Training stuff

    gates_passed = 0
    games_counter = 0
    bot_scores = []
    bot_score = sum(bot_scores)
    player_scores = []
    player_score = sum(player_scores)

    while(gates_passed < params["total_gates"]):
        # Check if the game has been ended by user
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Create a game, a player, and a gate
        game = Game(params["game_width"], params["game_height"]) # Width, Height
        #player = Player(params) 
        gate = Gate(params, game) 

        # Begin game and display (if applicable)
        #game.init_state(player, bot, gate)
        if params["display"]:
            game.render(player, bot, gate)

        games_counter += 1    
        # Run the game until a gate has been hit
        steps = 0
        while(steps < 1000):
            # Reset crash records
            player.crash = False
            bot.crash = False

            # If training, begin with high epsilon and work up, else use param epsilon
            if params["train"]:
                bot.epsilon = 1 - (gates_passed * params["epsilon_decay"])
            else:
                bot.epsilon = params["deploy_epsilon"]

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
            if(params["train"]):
                reward = bot.train_reward(bot_hit)
                bot.train_short_memory(bot.memory)
                bot.remember(state, bot_move, reward, new_state, bot_hit[0])
            if(player_hit[0] or bot_hit[0]):
                score(player, player_hit)
                score(bot, bot_hit)
                #break

            if params['display']:
                game.render(player, bot, gate)
                pygame.time.wait(params["speed"]) # Slows down game for viewing

            steps += 1
        print(f'Game {games_counter}\Bot: {bot.score}\Player: {player.score}\Steps: {steps}')
    model_weights = bot.state_dict()
    torch.save(model_weights, params["weights_path"])
    print("Weights saved")
    return

def score(user, gate_interact):
    if(gate_interact[1] == 1):
        user.score += 1
    elif(gate_interact[1] == 0):
        user.score -= 1
    elif(gate_interact[1] == -1):
        user.score -= 3

class Game:
    def __init__(self, width, height):
        pygame.display.set_caption("HaptyBird 2.0")
        self.game_width = width
        self.game_height = height
        self.surface = pygame.display.set_mode((width, height))
        self.background = pygame.image.load("imgs/background.jpg")
    
    def render(self, player, bot, gate):
        # UI display
        self.surface = pygame.display.set_mode((self.game_width, self.game_height)) # There has GOT to be a better way
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
        state = [
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
        ]

        # Convert state to 1s and 0s to help with training and make things easier
        state = [int(i) for i in state]
        return np.asarray(state)

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = init_params()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=30)
    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    params['speed'] = args.speed
    if params['train']:
        print("Training...")
        params['load_weights'] = True   
        run_games(params)
    if params['deploy']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        run_games(params)