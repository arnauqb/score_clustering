from typing import List
import numpy as np
from itertools import count

from .point import Point


class Cluster:
    """
    Represents a cluster of points.
    """
    _id = count()

    def __init__(self, points: List[Point]):
        self.id = next(self._id)
        self.points = np.array(points)
        self.subclusters = []
        self.score = self._get_score()
        self.centroid = self._get_centroid()

    def __getitem__(self, index):
        return self.points[index]

    def __len__(self):
        return len(self.points)

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __repr__(self):
        return "Cluster({})".format(self.points)

    def __eq__(self, other):
        return self.score == other.score

    def _get_score(self):
        return sum(point.score for point in self.points)

    def _get_centroid(self):
        return np.array(
            [
                np.mean([point.x for point in self.points]),
                np.mean([point.y for point in self.points]),
            ]
        )

