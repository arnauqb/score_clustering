import numpy as np
from typing import List
from copy import deepcopy
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sortedcontainers import SortedList

from .point import Point, Centroid 
from .cluster import Cluster


class ScoreClustering:
    def __init__(self, n_clusters, epsilon0=1):
        self.n_clusters = n_clusters
        self.epsilon0 = epsilon0

    def _compute_points_norm(self, points):
        norm_x = abs(
            max(point.x for point in points) - min(point.x for point in points)
        )
        norm_y = abs(
            max(point.y for point in points) - min(point.y for point in points)
        )
        return np.array([norm_x, norm_y])

    def fit(self, points: List[Point], niters=50):
        norm = self._compute_points_norm(points)
        avg_score = sum(point.score for point in points) / self.n_clusters
        centroids = self._get_initial_centroids(points, self.n_clusters)
        score = self.calculate_score_unbalance(centroids)
        best_score = score
        best_centroids = deepcopy(centroids)
        print(f"Initial score is {best_score}")
        for iter in range(niters):
            if score > 50:
                epsilon = self.epsilon0
            elif score > 10:
                epsilon = self.epsilon0 / 10
            elif score > 5:
                epsilon = self.epsilon0 / 100
            elif score > 2:
                epsilon = self.epsilon0 / 1000
            else:
                epsilon = self.epsilon0 / 10000
            self.update_centroids_positions(centroids, avg_score, norm, epsilon)
            self.update_centroids_scores(centroids, points)
            score = self.calculate_score_unbalance(centroids)
            print(score)
            if score < best_score:
                print(f"New best score: {score}")
                best_score = score
                best_centroids = deepcopy(centroids)
        return self.assign_points_to_closest_centroid(points, best_centroids)

    def update_centroids_positions(self, current_centroids, avg_score: float, norm, epsilon):
        centroids_k = [centroid.position for centroid in current_centroids]
        centroids_delaunay = Delaunay(centroids_k)
        for i, centroid in enumerate(current_centroids):
            neighbor_centroids = self._get_neighbor_clusters(
                centroids_delaunay, current_centroids, i
            )
            centroid.update_position(
                neighbor_centroids, avg_score, norm, epsilon
            )

    def assign_points_to_closest_centroid(self, points, centroids):
        centroids_k = [centroid.position for centroid in centroids]
        centroids_kdtree = KDTree(centroids_k)
        points_k = [[point.x, point.y] for point in points]
        distances, idcs = centroids_kdtree.query(points_k, k=1)
        clusters_points = [[] for _ in range(len(centroids))]
        for point, index in zip(points, idcs):
            clusters_points[index[0]].append(point)
        clusters = [Cluster(points) for points in clusters_points]
        return clusters

    def update_centroids_scores(self, centroids, points):
        for centroid in centroids:
            centroid.score = 0
        centroids_k = [centroid.position for centroid in centroids]
        centroids_kdtree = KDTree(centroids_k)
        points_k = [[point.x, point.y] for point in points]
        distances, idcs = centroids_kdtree.query(points_k, k=1)
        for point, index in zip(points, idcs):
            centroid = centroids[index[0]]
            centroid.score += point.score

    def calculate_score_unbalance(self, centroids):
        max_score = max([centroid.score for centroid in centroids if centroid.score > 0])
        min_score = min([centroid.score for centroid in centroids if centroid.score > 0])
        return max_score / min_score

    def _get_initial_centroids(self, points: List[Point], k: int):
        points_k = np.array([[point.x, point.y] for point in points])
        kmeans = KMeans(n_clusters=k).fit(points_k)
        centroid_positions = kmeans.cluster_centers_
        centroids = [Centroid(c[0], c[1]) for c in centroid_positions]
        self.update_centroids_scores(centroids, points)
        return centroids

    def _get_neighbor_clusters(self, delaunay, centroids, i):
        neighbors = list(
            set(
                indx
                for simplex in delaunay.simplices
                if i in simplex
                for indx in simplex
                if indx != i
            )
        )
        return [centroids[neighbor] for neighbor in neighbors if neighbor != -1]
