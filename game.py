import pygame
import torch
import torch.optim as opt
import numpy
import argparse
import distutils.util
from bot import HaptyBot
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
    params['total_gates'] = 250        
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    params['train'] = False # Do not change, use arguments in terminal to train
    params["deploy"] = True # Do not change, HaptyBot deployed to game should be default
    # Game parameters
    params["game_x_axis"] = 180000
    params["game_y_axis"] = 360000
    params["game_width"] = 180
    params["game_height"] = 360
    params["deploy_epsilon"] = 0.01
    params["cursor_y_axis"] = 10
    params["bot_start_x"] = 100
    params["player_start_x"] = 200
    params["gate_size"] = 100
    params["gate_min_distance"] = 100
    params["gate_cushion"] = 50
    # Data parameters
    params['weights_path'] = 'weights/weights.h5'
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'

    # User parameters

    return params

def run_games(params): # Runs the game
    pygame.init()
    bot = HaptyBot(params)
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
        player = Player(params) # Width (x axis)
        gate = Gate(params, game) # Width (x axis)

        # Begin game and display (if applicable)
        game.init_state(player, bot, gate)
        if params["display"]:
            game.render()

        # Run the game until a gate has been hit
        while(True):
            # Reset crash records
            player.crash = False
            bot.crash = False

            # If training, begin with high epsilon and work up, else use param epsilon
            if params["train"]:
                bot.epsilon = 1 - (gates_passed * params["epsilon_decay"])
            else:
                bot.epsilon = params["deploy_epsilon"]

            # Move gate downwards
            gate.update_y(game)

            # Move player/bot
            player.dosomething()
            bot.move()

            # Set scores and/or rewards as relevant
            if params["train"]:
                reward = bot.reward()
                bot.train_short_memory()
                bot.remember()
            if bot.y == gate.y and player.y == gate.y:
                score(bot)
                score(player)
                break

            record = get_record()
            if params['display']:
                game.render()
                pygame.time.wait(params["speed"]) # Slows down game for viewing

    return

class Game:
    def __init__(self, width, height):
        pygame.display.set_caption("HaptyBird 2.0")
        self.game_width = width
        self.game_height = height
        self.display = pygame.display.set_model((width, height))
        self.background = pygame.image.load("imgs/background.jpg")
    
    def render(self, player, bot, gate):
        # UI display
        font = pygame.font.SysFont("Segoe UI", 20)
        font_bold = pygame.font.SysFont("Segoe UI", 20, True)
        player_score = font_bold.render("PLAYER 1:", True, (255,255,255))
        player_score_n = font.render(str(player.score), True, (255,255,255))
        bot_score = font_bold.render("PLAYER 2:", True, (255,255,255))
        bot_score_n = font.render(str(bot.score), True, (255,255,255))

        self.display.blit(player_score, (5,10))
        self.display.blit(player_score_n, (15,10))
        self.display.blit(bot_score, (25,10))
        self.display.blit(bot_score_n, (35,10))
        self.display.blit(player_score, (45,10))
        # Game display
        pygame.draw.circle(self.display, "blue", (player.x, player.y), 1) # Player cursor
        pygame.draw.circle(self.display, "red", (bot.x, bot.y), 1) # Bot cursor
        gate.display() # Gate is complex, it has to draw itself

        pygame.display.update()
        pygame.event.get()

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = init_params()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=50)
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