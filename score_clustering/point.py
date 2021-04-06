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

    def score_vector(self, q: "Centroid", avg_score: float, epsilon: float = 1):
        score_diff = (q.score - self.score) / avg_score
        # distance = np.sign(score_diff) * (abs(score_diff) / (abs(score_diff) + epsilon))
        distance = score_diff  # max(0, score_diff)  # epsilon * np.tanh(1 - score_diff)
        direction = np.array([q.x - self.x, q.y - self.y])
        return distance * direction

    def update_position(self, points, avg_score, norm, epsilon=1):
        ret = np.zeros(2)
        for q in points:
            ret += self.score_vector(q, avg_score, epsilon)
        ret_norm = norm / np.linalg.norm(ret)
        ret = epsilon * ret * ret_norm
        self.x += ret[0]
        self.y += ret[1]

    @property
    def position(self):
        return np.array([self.x, self.y])
