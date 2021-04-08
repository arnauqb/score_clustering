import numpy as np
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
