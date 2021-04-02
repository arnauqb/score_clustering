from typing import List, Union
import numpy as np
from sortedcontainers import SortedSet, SortedList
from sortedcontainers.sortedlist import bisect_left
from sklearn.neighbors import KDTree

from .point import Point


class Cluster:
    """
    Represents a cluster of points.
    """

    def __init__(self, points: Union[List[Point], SortedSet]):
        if type(points) == SortedSet:
            self.points = points
        else:
            points_sorted = SortedSet(points)
            self.points = points_sorted
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

    def split(self):
        """
        Splits the cluster into two clusters of equal score.
        """
        half_score = sum(point.score for point in self.points) / 2
        points_cluster_1 = []
        score_1 = 0
        points_cluster_2 = []
        for point in self.points:
            if score_1 < half_score:
                points_cluster_1.append(point)
            else:
                points_cluster_2.append(point)
            score_1 += point.score

        cluster_1 = self.__class__(points_cluster_1)
        cluster_2 = self.__class__(points_cluster_2)
        self.subclusters = [cluster_1, cluster_2]
        return cluster_1, cluster_2


def merge(cluster1: Cluster, cluster2: Cluster):
    points = cluster1.points.union(cluster2.points)
    return Cluster(points)

def merge_and_split(clusters, cluster1: Cluster, cluster2: Cluster):
    clusters.remove(cluster1)
    clusters.remove(cluster2)
    cluster1, cluster2 = merge(cluster1, cluster2).split()
    clusters.add(cluster1)
    clusters.add(cluster2)



def _get_initial_split(cluster: Cluster, k: int):
    clusters = SortedList([cluster])
    for i in range(k - 1):
        biggest_cluster = clusters.pop(-1)
        cluster1, cluster2 = biggest_cluster.split()
        clusters.add(cluster1)
        clusters.add(cluster2)
    return clusters


def get_closest_clusters(kdtree, clusters, cluster):
    indices = kdtree.query(cluster.centroid.reshape(1,-1), k=len(clusters), return_distance=False)[0]
    ret = [clusters[idx] for idx in indices[1:]] # exclude own centroid
    return ret


def get_cluster_split(points: List[Point], k: int, niters: int = 10):
    cluster = Cluster(points)
    total_score = cluster.score
    avg_score = total_score / k
    clusters = _get_initial_split(cluster, k)
    changed = True
    while changed and niters:
        niters -= 1
        changed = False
        centroids = [cluster.centroid for cluster in clusters]
        centroids_kdtree = KDTree(np.array(centroids))
        split1 = None
        split2 = None
        for cluster in clusters:
            if cluster.score < avg_score:
                neighbor_clusters = get_closest_clusters(centroids_kdtree, clusters, cluster)
                for neighbor in neighbor_clusters:
                    if neighbor.score > avg_score:
                        changed = True
                        split1 = cluster
                        split2 = neighbor
                        break
            if changed:
                break
        if changed:
            merge_and_split(clusters, split1, split2)
    return clusters
