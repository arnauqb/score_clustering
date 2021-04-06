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
        cluster = Cluster(points)
        total_score = cluster.score
        avg_score = total_score / self.n_clusters
        clusters = self._get_initial_split(cluster, self.n_clusters)
        best_score = self.calculate_score_unbalance(clusters)
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
                    neighbor_clusters = self._get_neighbor_clusters(
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
                    self._merge_and_split(clusters, pair[0], pair[1])
                centroids = [cluster.centroid for cluster in clusters]
                clusters = self.assign_points_to_closest_centroid(points, centroids)
                score = self.calculate_score_unbalance(clusters)
                if score < best_score:
                    best_score = score
                    best_clusters = deepcopy(clusters)
                    print(f"Best score is {best_score}")
        return best_clusters

    def assign_points_to_closest_centroid(self, points, centroids):
        points_k = np.array([[point.x, point.y] for point in points])
        points_kdtree = KDTree(points_k)
        average_score = sum(point.score for point in points) / len(points)
        centroids_kdtree = KDTree(centroids)
        clusters_points = [[] for _ in range(len(centroids))]
        clusters_scores = np.zeros(len(centroids))
        points_assigned = np.zeros(len(points))
        for tolerance in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 50.0]: #, np.inf]:
            for i, centroid in enumerate(centroids):
                closest_points_idcs = points_kdtree.query(
                    centroid.reshape(1, -1), return_distance=False, k=len(points)
                )[0]
                for idx in closest_points_idcs:
                    if not points_assigned[idx]:
                        point = points[idx]
                        if clusters_scores[i] >= average_score * (1 + tolerance):
                            break
                        clusters_points[i].append(point)
                        clusters_scores[i] += point.score
                        points_assigned[idx] = 1
        #assert np.min(points_assigned) == 1
        for k, assigned in enumerate(points_assigned):
            if assigned:
                continue
            point_k = points_k[k]
            idx = centroids_kdtree.query(
                point_k.reshape(1, -1), return_distance=False, k=1
            )[0][0]
            clusters_points[idx].append(points[k])
        clusters = SortedList([Cluster(points) for points in clusters_points])
        return clusters

    def calculate_score_unbalance(self, clusters):
        max_score = max([cluster.score for cluster in clusters])
        min_score = min([cluster.score for cluster in clusters])
        return max_score / min_score

    def _merge_and_split(self, clusters, cluster1: Cluster, cluster2: Cluster):
        clusters.remove(cluster1)
        clusters.remove(cluster2)
        d1 = cluster1.centroid
        d2 = cluster2.centroid
        cluster1, cluster2 = cluster1.merge(cluster2).split(d1, d2)
        clusters.add(cluster1)
        clusters.add(cluster2)

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
