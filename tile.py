from pygame.sprite import Sprite
from pygame.rect import Rect

from utils import get_tile_pos

class Tile(Sprite):

    TILE_TYPE = 'TILE'

    def __init__(self, x, y, width, height, color, gap = 1):
        super().__init__()

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.gap = gap

        self.color = color  
        self.rect = None
        self.init()

    def init(self):
        y, x = get_tile_pos(self.y, self.x, self.width, self.height, self.gap)
        self.rect = Rect(x, y, self.width, self.height)

    def handle_collision(self, other):
        pass

    def update(self):
        pass

    def render(self, screen):
        screen.fill(self.color, self.rect)