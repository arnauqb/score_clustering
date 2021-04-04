from typing import List, Union
import numpy as np
from copy import deepcopy
from itertools import count
from scipy.spatial import Delaunay
from sortedcontainers import SortedList
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

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
    centroids_kdtree = KDTree(kmeans.cluster_centers_)
    clusters_points = [[] for _ in range(k)]
    for (point, point_k) in zip(points, points_k):
        idx = centroids_kdtree.query(point_k.reshape(1,-1), return_distance=False, k=1)[0][0]
        clusters_points[idx].append(point)
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
    neighbor_clusters = [clusters[neighbor] for neighbor in neighbors if neighbor != -1]
    neighbor_scores = [cluster.score for cluster in neighbor_clusters]
    return [neighbor_clusters[i] for i in np.argsort(neighbor_scores)[::-1]]

def calculate_score_unbalance(clusters):
    max_score = np.max([cluster.score for cluster in clusters])
    min_score = np.min([cluster.score for cluster in clusters])
    return max_score / min_score

def get_cluster_split(points: List[Point], k: int, niters: int = 10):
    cluster = Cluster(points)
    total_score = cluster.score
    avg_score = total_score / k
    clusters = _get_initial_split(cluster, k)
    best_score = calculate_score_unbalance(clusters)
    best_clusters = deepcopy(clusters)
    while niters:
        niters -= 1
        centroids = [cluster.centroid for cluster in clusters]
        centroids_delaunay = Delaunay(np.array(centroids))
        to_change = set()
        to_join_pairs = []
        for (i, cluster) in enumerate(clusters):
            if cluster.id in to_change:
                continue
            if cluster.score < avg_score:
                neighbor_clusters = get_closest_clusters(
                    centroids_delaunay, clusters, i
                )
                for neighbor in neighbor_clusters:
                    if neighbor.id in to_change:
                        continue
                    if neighbor.score > avg_score:
                        to_join_pairs.append((cluster, neighbor))
                        to_change.add(cluster.id)
                        to_change.add(neighbor.id)
                        break
        if to_change:
            for pair in to_join_pairs:
                merge_and_split(clusters, pair[0], pair[1])
        score = calculate_score_unbalance(clusters)
        if score < best_score:
            best_score = score
            best_clusters = deepcopy(clusters)
            print(f"Best score is {best_score}")
    return best_clusters
