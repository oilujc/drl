from pygame.sprite import Sprite, Group
from tile import Tile
from utils import get_tile_size

class Map(Sprite):
    def __init__(self, width, height, row = 6, col = 8, gap = 1):
        super().__init__()
        self.row = row
        self.col = col
        self.gap = gap
        
        self.max_width = width
        self.max_height = height
        self.tiles = Group()

        self.init()

    def init(self):

        for r in range(self.row):
            for c in range(self.col):

                tile_width, tile_height = get_tile_size(self.max_width, self.max_height, self.row, self.col, self.gap)

                tile = Tile(c, r, tile_width, tile_height, (255, 255, 255), self.gap)
                self.tiles.add(tile)

    def update(self):
        pass

    def render(self, screen):

        for i in range(self.row):
            for j in range(self.col):
                tile = self.tiles.sprites()[i * self.col + j]
                screen.fill(tile.color, tile.rect)
