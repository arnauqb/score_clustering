import numpy as np
from typing import List
from copy import deepcopy
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sortedcontainers import SortedList

from .point import Point
from .cluster import Cluster


class ScoreClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, points: List[Point], niters=50):
        pass

    def assign_points_to_closest_centroid(self, points, centroids):
        pass

    def calculate_score_unbalance(self, clusters):
        max_score = max([cluster.score for cluster in clusters])
        min_score = min([cluster.score for cluster in clusters])
        return max_score / min_score

    def _get_initial_split(self, cluster: Cluster, k: int):
        points = cluster.points
        points_k = np.array([[point.x, point.y] for point in points])
        kmeans = KMeans(n_clusters=k).fit(points_k)
        return self.assign_points_to_closest_centroid(points, kmeans.cluster_centers_)

    def _get_neighbor_clusters(self, delaunay, clusters, i):
        neighbors = list(
            set(
                indx
                for simplex in delaunay.simplices
                if i in simplex
                for indx in simplex
                if indx != i
            )
        )
        neighbor_clusters = [
            clusters[neighbor] for neighbor in neighbors if neighbor != -1
        ]
        neighbor_scores = [cluster.score for cluster in neighbor_clusters]
        return [neighbor_clusters[i] for i in np.argsort(neighbor_scores)[::-1]]
