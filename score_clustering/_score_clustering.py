import numpy as np
from typing import List
from copy import deepcopy
from random import sample
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sortedcontainers import SortedList, SortedSet

from .point import Point, Centroid
from .cluster import Cluster


class ScoreClustering:
    def __init__(self, n_clusters, epsilon0):
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

    #def fit(self, points: List[Point], niter=50):
    #    clusters = self._get_initial_split(points)
    #    avg_score = sum(point.score for point in points) / self.n_clusters
    #    #best_score = self.calculate_score_unbalance(clusters)
    #    #best_clusters = deepcopy(clusters)
    #    while niter:
    #        niter -= 1
    #        to_change = set()
    #        to_join_pairs = []
    #        #for (i, cluster) in enumerate(clusters):
    #        for i in sample(range(self.n_clusters), self.n_clusters):
    #            cluster = clusters[i]
    #            if cluster.id in to_change:
    #                continue
    #            if cluster.score < avg_score:
    #                neighbor_clusters = self._get_neighbor_clusters(cluster)
    #                for neighbor in reversed(neighbor_clusters):
    #                    if neighbor.id in to_change:
    #                        continue
    #                    if neighbor.score > avg_score:
    #                        to_join_pairs.append((cluster, neighbor))
    #                        to_change.add(cluster.id)
    #                        to_change.add(neighbor.id)
    #                        break
    #        if to_change:
    #            for pair in to_join_pairs:
    #                self._merge_and_split(clusters, pair[0], pair[1])
    #            #centroids = [cluster.centroid for cluster in clusters]
    #            #clusters = self.assign_points_to_closest_centroid(points, centroids)
    #            score = self.calculate_score_unbalance(clusters)
    #            print(score)
    #            #if score < best_score:
    #            #    best_score = score
    #            #    best_clusters = deepcopy(clusters)
    #            #    print(f"Best score is {best_score}")
    #    return clusters# best_clusters
    
    def fit(self, points: List[Point], niter=50):
        norm = self._compute_points_norm(points)
        avg_score = sum(point.score for point in points) / self.n_clusters
        centroids = self._get_initial_centroids(points, self.n_clusters)
        best_centroids = deepcopy(centroids)
        clusters = self.assign_points_to_closest_centroid(points, best_centroids)
        score = self.calculate_score_unbalance(clusters)
        best_score = score
        print(f"Initial score is {best_score}")
        for iter in range(niter):
            epsilon = self.epsilon0
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
            self.update_centroids_positions(centroids, clusters, avg_score, norm, epsilon)
            #self.update_centroids_scores(centroids, points)
            clusters = self.assign_points_to_closest_centroid(points, centroids)
            score = self.calculate_score_unbalance(clusters)
            print(score)
            if score < best_score:
                print(f"New best score: {score}")
                best_score = score
                best_centroids = deepcopy(centroids)
        return self.assign_points_to_closest_centroid(points, best_centroids)

    def calculate_score_unbalance(self, clusters):
        max_score = max([cluster.score for cluster in clusters if len(cluster.points) > 0])
        min_score = min([cluster.score for cluster in clusters if len(cluster.points) > 0])
        return max_score / min_score

    #def _merge_and_split(self, clusters, cluster1: Cluster, cluster2: Cluster):
    #    clusters.remove(cluster1)
    #    clusters.remove(cluster2)
    #    cluster1, cluster2 = cluster1.merge(cluster2).split()
    #    clusters.add(cluster1)
    #    clusters.add(cluster2)

    #def _get_initial_split(self, points: List[Point]):
    #    cluster = Cluster(points)
    #    clusters = SortedList([cluster])
    #    for _ in range(self.n_clusters):
    #        biggest_cluster = clusters.pop(-1)
    #        c1, c2 = biggest_cluster.split()
    #        clusters.update([c1, c2])
    #    return clusters

    def _get_neighbor_clusters(self, cluster):
        neighbor_clusters = [neighbor.cluster for point in cluster.points for neighbor in point.neighbors]
        neighbor_clusters = SortedSet(neighbor_clusters)
        neighbor_clusters.discard(cluster)
        return neighbor_clusters


    def update_centroids_positions(self, current_centroids, clusters, avg_score: float, norm, epsilon):
        for i, centroid in enumerate(current_centroids):
            neighbor_clusters = self._get_neighbor_clusters(
                clusters[i]
            )
            neighbor_centroids = [cluster.centroid for cluster in neighbor_clusters]
            centroid.update_position(
                neighbor_centroids, avg_score, norm, epsilon
            )

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
        clusters = [Cluster(points, centroid) for points, centroid in zip(clusters_points, centroids)]
        return clusters

    #def update_centroids_scores(self, centroids, points):
    #    for centroid in centroids:
    #        centroid.score = 0
    #    centroids_k = [centroid.position for centroid in centroids]
    #    centroids_kdtree = KDTree(centroids_k)
    #    points_k = [[point.x, point.y] for point in points]
    #    distances, idcs = centroids_kdtree.query(points_k, k=1)
    #    for point, index in zip(points, idcs):
    #        centroid = centroids[index[0]]
    #        centroid.score += point.score

    def _get_initial_centroids(self, points: List[Point], k: int):
        points_k = np.array([[point.x, point.y] for point in points])
        kmeans = KMeans(n_clusters=k).fit(points_k)
        centroid_positions = kmeans.cluster_centers_
        centroids = [Centroid(c[0], c[1]) for c in centroid_positions]
        #self.update_centroids_scores(centroids, points)
        return centroids

                    
