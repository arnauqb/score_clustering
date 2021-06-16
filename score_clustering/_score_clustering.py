import numpy as np
from typing import List
from copy import deepcopy
from random import sample, randint
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sortedcontainers import SortedList, SortedSet

from .point import Point
from .cluster import Cluster


class ScoreClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, points: List[Point], maxiter=50000, target_score=1.05):
        clusters = self._get_initial_clusters(points)
        avg_score = sum(point.score for point in points) / self.n_clusters
        score = self.calculate_score_unbalance(clusters)
        while score > target_score and maxiter > 0:
            maxiter -= 1
            for point in sample(points, len(points)):
                point_cluster = point.cluster
                if point_cluster.score < avg_score:
                    for neighbor in sample(point.neighbors, len(point.neighbors)):
                        if neighbor.cluster == point_cluster:
                            continue
                        if neighbor.cluster.is_articulation_point(
                            neighbor
                        ):  # neighbor in neighbor.cluster.articulation_points:
                            continue
                        if neighbor.cluster.score > avg_score:
                            if point.cluster.score + neighbor.score <= 1.1 * avg_score:
                                self._transfer_point(neighbor, point)
                                break
            self._reassign_isolated_points(points)
            score = self.calculate_score_unbalance(clusters)
        return clusters  # best_clusters

    def _transfer_point(self, outgoing, incoming):
        outgoing.cluster.remove(outgoing)
        incoming.cluster.add(outgoing)

    def _get_neighbor_clusters(self, cluster):
        neighbor_clusters = [
            neighbor.cluster for point in cluster.points for neighbor in point.neighbors
        ]
        neighbor_clusters = SortedSet(neighbor_clusters)
        neighbor_clusters.discard(cluster)
        return neighbor_clusters

    def assign_points_to_closest_centroid(self, points, centroids):
        centroids_k = [centroid.position for centroid in centroids]
        centroids_kdtree = KDTree(centroids_k)
        points_k = [[point.x, point.y] for point in points]
        distances, idcs = centroids_kdtree.query(points_k, k=1)
        clusters_points = [[] for _ in range(len(centroids))]
        for centroid in centroids:
            centroid.score = 0
        for point, index in zip(points, idcs):
            centroids[index[0]].score += point.score
            clusters_points[index[0]].append(point)
        clusters = [
            Cluster(points, centroid)
            for points, centroid in zip(clusters_points, centroids)
        ]
        return clusters

    def _get_initial_clusters(self, points: List[Point]):
        points_k = np.array([[point.x, point.y] for point in points])
        best_centroids = None
        best_score = np.inf
        for _ in range(10):
            kmeans = KMeans(n_clusters=self.n_clusters).fit(points_k)
            centroid_positions = kmeans.cluster_centers_
            centroids_kdtree = KDTree(centroid_positions)
            distances, idcs = centroids_kdtree.query(points_k, k=1)
            clusters_points = [[] for _ in range(self.n_clusters)]
            for point, index in zip(points, idcs):
                clusters_points[index[0]].append(point)
            clusters = [Cluster(points) for points in clusters_points]
            score = self.calculate_score_unbalance(clusters)
            if score < best_score:
                best_score = score
                best_centroids = deepcopy(centroid_positions)
        print(f"Best initial score is {best_score}")
        centroids_kdtree = KDTree(best_centroids)
        distances, idcs = centroids_kdtree.query(points_k, k=1)
        clusters_points = [[] for _ in range(self.n_clusters)]
        for point, index in zip(points, idcs):
            clusters_points[index[0]].append(point)
        clusters = [Cluster(ps) for ps in clusters_points]
        for point in points:
            assert point.cluster is not None
        self._reassign_isolated_points(points)
        return clusters

    def _reassign_isolated_points(self, points):
        isolated_points = []
        for point in points:
            isolated = True
            if point.neighbors:
                for neighbor in point.neighbors:
                    if neighbor.cluster == point.cluster:
                        isolated = False
                        break
                if isolated:
                    isolated_points.append(point)
        for point in isolated_points:
            for neighbor in sample(point.neighbors, len(point.neighbors)):
                if neighbor.cluster is not None:
                    point.cluster.remove(point)
                    neighbor.cluster.add(point)

    def calculate_score_unbalance(self, clusters):
        max_score = max([cluster.score for cluster in clusters])
        min_score = min([cluster.score for cluster in clusters])
        return max_score / min_score
