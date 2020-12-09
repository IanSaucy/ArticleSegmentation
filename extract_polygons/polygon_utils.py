from random import randrange
from typing import Tuple


def random_color() -> Tuple[int, int, int]:
    """
    Simple function that generates a random RGB color
    Returns:
        A tuple representing a random RGB color
    """
    return randrange(0, 255), randrange(0, 255), randrange(0, 255)
