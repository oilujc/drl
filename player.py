from config import Config
from tile import Tile
from utils import get_tile_index, get_tile_pos

class Player(Tile):

    TILE_TYPE = 'PLAYER'

    PLAYER_UP = 0
    PLAYER_DOWN = 1
    PLAYER_LEFT = 2
    PLAYER_RIGHT = 3
    PLAYER_TAKE = 4
    PLAYER_DROP = 5

    PLAYER_NUM_ACTIONS = 6

    PLAYER_VALID_COLLISIONS_WITH = [
        'TARGET',
        'BOX',
    ]

    PLAYER_BASE_STATE = 1
    PLAYER_TAKE_BOX_STATE = 2
    PLAYER_DROP_BOX_STATE = 3
    PLAYER_GET_PLAYER_REWARD_STATE = 4

    PLAYER_COLLIDE_WITH_NOTHING = 0
    PLAYER_COLLIDE_WITH_BOX = 1
    PLAYER_COLLIDE_WITH_TARGET = 2

    ACTIONS = {
        0: 'UP',
        1: 'DOWN',
        2: 'LEFT',
        3: 'RIGHT',
        4: 'TAKE',
        5: 'DROP',
    }

    STATES = {
        PLAYER_BASE_STATE: 'BASE',
        PLAYER_TAKE_BOX_STATE: 'TAKE BOX',
        PLAYER_DROP_BOX_STATE: 'DROP BOX',
        PLAYER_GET_PLAYER_REWARD_STATE: 'GET REWARD',
    }

    def __init__(self, x, y, width, height, color = (255, 0 , 0), gap = 1):
        super().__init__(x, y, width, height, color, gap)

        self.is_collided_with = self.PLAYER_COLLIDE_WITH_NOTHING
        self.current_state = self.PLAYER_BASE_STATE


    def handle_collision(self, other):
        if self.rect.colliderect(other.rect) and other.TILE_TYPE in self.PLAYER_VALID_COLLISIONS_WITH:
            self.is_collided_with = getattr(self, 'PLAYER_COLLIDE_WITH_' + other.TILE_TYPE)
            return
    
        self.is_collided_with = self.PLAYER_COLLIDE_WITH_NOTHING

    def update(self, action):

        x, y = self.x, self.y

        if action == self.PLAYER_UP and y > 0:
            y -= 1
        elif action == self.PLAYER_DOWN and y < Config.ROW - 1:
            y += 1
        elif action == self.PLAYER_LEFT and x > 0:
            x -= 1
        elif action == self.PLAYER_RIGHT and x < Config.COL - 1:
            x += 1

        elif action == self.PLAYER_TAKE and self.is_collided_with == self.PLAYER_COLLIDE_WITH_BOX and self.current_state == self.PLAYER_BASE_STATE:
            self.current_state = self.PLAYER_TAKE_BOX_STATE

        elif action == self.PLAYER_DROP and self.current_state == self.PLAYER_TAKE_BOX_STATE and self.is_collided_with == self.PLAYER_COLLIDE_WITH_NOTHING:
            self.current_state = self.PLAYER_DROP_BOX_STATE

        elif action == self.PLAYER_DROP and self.current_state == self.PLAYER_TAKE_BOX_STATE and self.is_collided_with == self.PLAYER_COLLIDE_WITH_TARGET:
            self.current_state = self.PLAYER_GET_PLAYER_REWARD_STATE

        pos_y, pos_x = get_tile_pos(y, x, self.width, self.height, self.gap)

        self.y, self.x = y, x
        self.rect.y, self.rect.x = pos_y, pos_x



        

