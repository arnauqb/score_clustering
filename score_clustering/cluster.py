from typing import List
import numpy as np
from scipy.spatial import distance_matrix
from itertools import count

from .point import Point, sort_points


class Cluster:
    """
    Represents a cluster of points.
    """

    _id = count()

    def __init__(self, points: List[Point], centroid):
        self.id = next(self._id)
        self.points = np.array(points)
        for point in points:
            point.cluster = self
        self.score = self._get_score()
        self.centroid = centroid #self._get_centroid()

    def __getitem__(self, index):
        return self.points[index]

    def __len__(self):
        return len(self.points)

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return "Cluster({})".format(self.points)

    def __eq__(self, other):
        return self.score == other.score

    def _get_score(self):
        return sum(point.score for point in self.points)

    def _get_diametric_points(self):
        points_k = [[point.x, point.y] for point in self.points]
        dmatrix = distance_matrix(points_k, points_k)
        idx1, idx2 = np.unravel_index(dmatrix.argmax(), dmatrix.shape)
        return [points_k[idx] for idx in [idx1, idx2]]

    def merge(self, other: "Cluster"):
        points = np.concatenate((self.points, other.points))
        return self.__class__(points)

    def split(self):
        """
        Splits the cluster into two clusters of equal score.
        """
        d1, d2 = self._get_diametric_points()
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
        return cluster_1, cluster_2

    def _get_centroid(self):
        return np.array(
            [
                np.mean([point.x for point in self.points]),
                np.mean([point.y for point in self.points]),
            ]
        )
