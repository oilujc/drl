from tile import Tile

class Box(Tile):

    TILE_TYPE = 'BOX'

    def __init__(self, x, y, width, height, color = (0, 0, 255), gap = 1):
        super().__init__(x, y, width, height, color, gap)

    def handle_collision(self, other):
        pass

    def update(self):
        pass

    def render(self, screen):
        super().render(screen)