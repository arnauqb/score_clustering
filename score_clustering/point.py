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

class Centroid:
    def __init__(self, x, y, score = 0):
        self.x = x
        self.y = y
        self.score = score

    def get_weight(self, q: "Centroid", avg_score):
        return (q.score - self.score)# * abs(self.score - avg_score) / avg_score 

    def score_vector(self, q: "Centroid", avg_score: float):
        weight = self.get_weight(q, avg_score)
        direction = np.array([q.x - self.x, q.y - self.y])
        return weight * direction

    def update_position(self, points, avg_score, norm, epsilon=1):
        ret = np.zeros(2)
        #weights = [self.get_weight(q, avg_score) for q in points]
        #max_arg = np.argmax(weights)
        #q = points[max_arg]
        #direction = np.array([q.x - self.x, q.y - self.y])
        #ret = epsilon * direction * norm / np.linalg.norm(direction)
        for q in points:
            ret += self.score_vector(q, avg_score)
        direction_norm = np.linalg.norm(ret)
        if direction_norm == 0:
            return
        ret_norm = norm / direction_norm
        ret = epsilon / 2 * ret * ret_norm
        self.x += ret[0]
        self.y += ret[1]

    @property
    def position(self):
        return np.array([self.x, self.y])

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

