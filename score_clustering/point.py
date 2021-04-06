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


def score_vector(p: Point, q: Point, avg_score: float):
    score_diff = (q.score - p.score) / avg_score
    return (
        np.sign(score_diff)
        * np.log(1 + abs(score_diff))
        * np.array([q.x - p.x, q.y - p.y])
    )


def calculate_centroid_update(p: Point, points: List[Point], avg_score: float):
    ret = np.zeros(2)
    for q in points:
        ret += score_vector(p, q, avg_score)
    return ret
