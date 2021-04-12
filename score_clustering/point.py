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

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)

    #def __deepcopy__(self, memo):
    #    cls = self.__class__
    #    result = cls.__new__(cls)
    #    memo[id(self)] = result
    #    for k, v in self.__dict__.items():
    #        if k == "cluster" or k == "neighbors":
    #            continue
    #        setattr(result, k, deepcopy(v, memo))
    #    return result

def get_perpendicular_vector(d):
    if d[1] == 0:
        dp = [-d[1] / d[0], 1]
    else:
        dp = [1, -d[0] / d[1]]
    return dp


def get_intersection_time(d1, d2, p):
    """
    Given the line that joins d1 -- d2, finds the intersection time
    along d1 -- d2 to intersect a line perpendicular to d1 -- d2 that contains p.
    """
    d = [d2[0] - d1[0], d2[1] - d1[1]]
    d /= np.linalg.norm(d)
    dp = get_perpendicular_vector(d)

    A = np.array([[dp[0], -d[0]], [dp[1], -d[1]]])
    b = [d1[0] - p.x, d1[1] - p.y]
    x = np.linalg.solve(A, b)
    return x[1]


def sort_points(points, d1, d2):
    intersection_times = list(map(lambda x: get_intersection_time(d1, d2, x), points))
    idx = np.argsort(intersection_times)
    return points[idx]

