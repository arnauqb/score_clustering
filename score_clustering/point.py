import numpy as np
from copy import deepcopy
from typing import List, Optional


class Point:
    """
    A class that represents a point.
    """

    def __init__(
        self, x: float, y: float, score: float = 0, name: Optional[str] = None
    ):
        self.x = x
        self.y = y
        self.score = score
        self.name = name
        self.neighbors = None
        self.cluster = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)
