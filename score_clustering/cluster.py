from typing import List, Union
import numpy as np
from scipy.spatial import Delaunay
from sortedcontainers import SortedList
from sklearn.cluster import KMeans

from .point import Point, sort_points


class Cluster:
    """
    Represents a cluster of points.
    """

    def __init__(self, points: List[Point]):
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


def merge(cluster1: Cluster, cluster2: Cluster):
    points = np.concatenate((cluster1.points, cluster2.points))
    return Cluster(points)


def merge_and_split(clusters, cluster1: Cluster, cluster2: Cluster):
    clusters.remove(cluster1)
    clusters.remove(cluster2)
    d1 = cluster1.centroid
    d2 = cluster2.centroid
    cluster1, cluster2 = merge(cluster1, cluster2).split(d1, d2)
    clusters.add(cluster1)
    clusters.add(cluster2)


def _get_initial_split(cluster: Cluster, k: int):
    points = cluster.points
    points_k = np.array([[point.x, point.y] for point in points])
    kmeans = KMeans(n_clusters=k).fit(points_k)
    clusters_points = [[] for _ in range(k)]
    for point, label in zip(points, kmeans.labels_):
        clusters_points[label].append(point)
    clusters = SortedList([Cluster(points) for points in clusters_points])
    return clusters

def get_closest_clusters(delaunay, clusters, i):
    neighbors = list(
        set(
            indx
            for simplex in delaunay.simplices
            if i in simplex
            for indx in simplex
            if indx != i
        )
    )
    return [clusters[neighbor] for neighbor in neighbors if neighbor != -1]


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
        centroids_delaunay = Delaunay(np.array(centroids))
        split1 = None
        split2 = None
        for (i, cluster) in enumerate(clusters):
            if cluster.score < avg_score:
                neighbor_clusters = get_closest_clusters(
                    centroids_delaunay, clusters, i
                )
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
