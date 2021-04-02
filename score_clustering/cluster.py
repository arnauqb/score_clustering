from typing import List
from sortedcontainers import SortedSet

from .point import Point

class Cluster:
    """
    Represents a cluster of points.
    """
    def __init__(self, points: List[Point]):
        points_sorted = SortedSet(points)
        self.points = points_sorted
        self.subclusters = []

    def __getitem__(self, index):
        return self.points[index]

    def __len__(self):
        return len(self.points)

    def __eq__(self, other):
        for p, q in zip(self.points, other.points):
            if p != q:
                return False
        return True

    def split(self):
        """
        Splits the cluster into two clusters of equal score.
        """
        half_score = sum(point.score for point in self.points) / 2
        points_cluster_1 = []
        score_1 = 0
        points_cluster_2 = []
        for point in self.points:
            print(point.x, point.y, point.score)
            if score_1 < half_score:
                points_cluster_1.append(point)
            else:
                points_cluster_2.append(point)
            score_1 += point.score

        cluster_1 = self.__class__(points_cluster_1)
        cluster_2 = self.__class__(points_cluster_2)
        self.subclusters = [cluster_1, cluster_2]
        return cluster_1, cluster_2


