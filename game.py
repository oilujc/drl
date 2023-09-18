import pygame, random
from base.game_base import GameBase 
from config import Config

from map import Map
from player import Player
from box import Box
from target import Target

from utils import get_tile_size
import numpy as np
import logging

logging.basicConfig(filename='logs/game.log', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('game')

class Game(GameBase):


    def __init__(self):

        self.screen = None
        self.running = True

        self.reward = Config.GAME_EVENTS_REWARDS[Player.PLAYER_BASE_STATE]

        self.map = None
        self.time = None

        self.player = None
        self.action = None
        self.sprites = None
        self.font = None

        self.current_time = 0
        self.current_score = 0

        self.current_total_player_games = 0
        self.num_box_taken = 0
        
        self.init()

    def init(self):

        pygame.init()
        self.screen = pygame.display.set_mode(Config.SCREEN_SIZE)
        self.time = pygame.time.Clock()

        self.map = Map(Config.MAP_WIDTH, Config.MAP_HEIGHT, Config.ROW, Config.COL, Config.GAP)
        self.sprites = pygame.sprite.Group()
        self.font = pygame.font.SysFont('Arial', 30)

        self.reset()

    def gen_player(self):
        
        player_pos = random.choice([i for i in range(Config.ROW * Config.COL)])
        player_x, player_y = player_pos % Config.COL, player_pos // Config.COL

        player_width, player_height = get_tile_size(Config.MAP_WIDTH, Config.MAP_HEIGHT, Config.ROW, Config.COL, Config.GAP)
        
        self.player = Player(player_x, player_y, player_width, player_height, Config.PLAYER_COLOR, Config.GAP)

    def gen_box(self, in_player_pos = False):

        if not in_player_pos:
            box_pos = random.choice([i for i in range(Config.ROW * Config.COL)])
            box_x, box_y = box_pos % Config.COL, box_pos // Config.COL
        else:
            box_x, box_y = self.player.x, self.player.y

        if box_x == self.sprites.sprites()[0].x and box_y == self.sprites.sprites()[0].y and not in_player_pos:
            self.gen_box()
            return

        box_width, box_height = get_tile_size(Config.MAP_WIDTH, Config.MAP_HEIGHT, Config.ROW, Config.COL, Config.GAP)
        box = Box(box_x, box_y, box_width, box_height, Config.BOX_COLOR, Config.GAP)

        self.sprites.add(box)

    def gen_random_target(self):

        target_pos = random.choice([i for i in range(Config.ROW * Config.COL)])
        target_x, target_y = target_pos % Config.COL, target_pos // Config.COL

        target_width, target_height = get_tile_size(Config.MAP_WIDTH, Config.MAP_HEIGHT, Config.ROW, Config.COL, Config.GAP)
        target = Target(target_x, target_y, target_width, target_height, Config.TARGET_COLOR, Config.GAP)

        self.sprites.add(target)

    def handle_game_events(self):

        if self.player.current_state == self.player.PLAYER_GET_PLAYER_REWARD_STATE:
            self.current_total_player_games += 1
        
        elif self.player.current_state == self.player.PLAYER_TAKE_BOX_STATE:
            for i in range(len(self.sprites)):
                if self.sprites.sprites()[i].TILE_TYPE == 'BOX':
                    self.sprites.remove(self.sprites.sprites()[i])
                    self.num_box_taken += 1

                    break 
                
            return
        
        elif self.player.current_state == self.player.PLAYER_DROP_BOX_STATE:
            self.gen_box([self.player.x, self.player.y])
            return


    def reset(self):
        
        self.num_box_taken = 0
        self.sprites.empty()

        self.gen_player()
        self.gen_random_target()
        self.gen_box()

    def get_state(self):

        """ states for q-learning
            state = [
                [[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy]],
                [[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy]],
                [[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy]],
                [[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy]],
                [[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy]],
                [[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy],[player_xy, box_xy, target_xy]],
            ], time, score, total_moves, game_event_type, player_state
        
        """

        state = np.zeros((Config.ROW, Config.COL, 3))
        target_xy = [self.sprites.sprites()[0].x, self.sprites.sprites()[0].y]
        box_xy = [self.player.x, self.player.y]
        player_xy = [self.player.x, self.player.y]

        if len(self.sprites) > 1:
            box_xy = [self.sprites.sprites()[1].x, self.sprites.sprites()[1].y]
        
        state[target_xy[1]][target_xy[0]][0] = 1
        state[box_xy[1]][box_xy[0]][1] = 2
        state[player_xy[1]][player_xy[0]][2] = 3

        game_state = np.array([self.player.current_state, self.player.is_collided_with, self.num_box_taken], dtype=np.float32)

        return np.concatenate((state.flatten(), game_state), dtype=np.float32)

    def step(self, action, state):

        is_done = False 
        reward = self.reward
        
        is_done = False if self.player.current_state != self.player.PLAYER_GET_PLAYER_REWARD_STATE else True
        next_state = self.get_state()

        print('-------------------')
        print('Action: ', self.player.ACTIONS[action])
        print('State: ', state)
        print('Reward: ', reward)
        print('Is done: ', is_done)
        
        return state, action, reward, next_state, is_done
    
    def get_reward(self):

        self.reward = Config.GAME_EVENTS_REWARDS[self.player.current_state]

        if self.player.current_state == self.player.PLAYER_DROP_BOX_STATE:
            self.reward = self.reward + (self.num_box_taken * self.reward)

        if self.player.current_state != self.player.PLAYER_BASE_STATE:
            logger.info(f'Action: {self.player.STATES[self.player.current_state]} Reward: {self.reward} Num box taken: {self.num_box_taken}')
    
        

    def update(self, action = None):

        if self.player.current_state != self.player.PLAYER_TAKE_BOX_STATE:
            self.player.current_state = self.player.PLAYER_BASE_STATE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                    
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                elif event.key == pygame.K_SPACE:
                    if self.player.current_state == self.player.PLAYER_TAKE_BOX_STATE:
                        action = 5
                    else:
                        action = 4

        for sprite in self.sprites:
            self.player.handle_collision(sprite)

        if action != None:
            self.player.update(action)  
    
        self.current_time += self.time.get_time() / 1000

        self.handle_game_events()
        self.get_reward()
       
        action = None
      
    def render(self, screen):
        
        screen.fill((0, 0, 0))

        self.map.render(screen)

        for sprite in self.sprites:
            screen.fill(sprite.color, sprite.rect)
            
        self.player.render(screen)

        total_player_games = self.font.render(f'Total games: {self.current_total_player_games}', True, (255, 255, 255))
        screen.blit(total_player_games, (0, Config.MAP_HEIGHT + 10))
        
        pygame.display.flip()


    def quit(self):
        pygame.quit()
        quit()
    