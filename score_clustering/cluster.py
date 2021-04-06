from typing import List
import numpy as np
from itertools import count

from .point import Point, sort_points


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

    def split(self, d1, d2):
        """
        Splits the cluster into two clusters of equal score.
        """
        half_score = sum(point.score for point in self.points) / 2
        points_cluster_1 = []
        score_1 = 0
        points_cluster_2 = []
        points_sorted = sort_points(self.points, d1, d2)
        for point in points_sorted:
            if score_1 < half_score:
                points_cluster_1.append(point)
            else:
                points_cluster_2.append(point)
            score_1 += point.score

        cluster_1 = self.__class__(points_cluster_1)
        cluster_2 = self.__class__(points_cluster_2)
        self.subclusters = [cluster_1, cluster_2]
        return cluster_1, cluster_2

    def merge(self, other: "Cluster"):
        points = np.concatenate((self.points, other.points))
        return self.__class__(points)


