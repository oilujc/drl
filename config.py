class Config:
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600

    MAP_HEIGHT = SCREEN_HEIGHT - 100
    MAP_WIDTH = SCREEN_WIDTH

    SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)

    FPS = 60

    COL = 8
    ROW = 6
    GAP = 1

    GAME_EVENTS_REWARDS = {
        1: -1,
        2: 100,
        3: 100 * -1,
        4: 1000,
    }

    MAX_ITER = 1000

    PLAYER_COLOR = (255, 0, 0)
    BOX_COLOR = (0, 255, 0)
    TARGET_COLOR = (0, 0, 255)

    LR = 0.001
    GAMMA = 0.9
    EPSILON = 0.9
    HIDDEN_SIZE = 50
    BATCH_SIZE = 32