def get_tile_size(width, height, row, col, gap):
    """
    Calculate the size of tile based on the given parameters

    Args:
        width (int): the width of the map
        height (int): the height of the map
        row (int): the number of rows
        col (int): the number of columns
        gap (int): the gap between tiles
    
    """


    tile_width = (width - gap * (col - 1)) / col
    tile_height = (height - gap * (row - 1)) / row

    return tile_width, tile_height

def get_tile_pos(row, col, tile_width, tile_height, gap):
    """
    Calculate the position of tile based on the given parameters

    Args:
        row (int): the row of the tile
        col (int): the column of the tile
        tile_width (int): the width of the tile
        tile_height (int): the height of the tile
        gap (int): the gap between tiles
    
    """



    return row * (tile_height + gap), col * (tile_width + gap)

def get_tile_index(x, y, tile_width, tile_height, gap):
    """
    Calculate the index of tile based on the given parameters

    Args:
        x (int): the x coordinate of the tile
        y (int): the y coordinate of the tile
        tile_width (int): the width of the tile
        tile_height (int): the height of the tile
        gap (int): the gap between tiles
    
    """


    return x // (tile_width + gap), y // (tile_height + gap)