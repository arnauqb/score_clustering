import numpy as np
from typing import List
from copy import deepcopy
from random import sample
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sortedcontainers import SortedList, SortedSet

from .point import Point
from .cluster import Cluster


class ScoreClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, points: List[Point], niter=50):
        clusters = self._get_initial_split(points)
        avg_score = sum(point.score for point in points) / self.n_clusters
        #best_score = self.calculate_score_unbalance(clusters)
        #best_clusters = deepcopy(clusters)
        while niter:
            niter -= 1
            to_change = set()
            to_join_pairs = []
            #for (i, cluster) in enumerate(clusters):
            for i in sample(range(self.n_clusters), self.n_clusters):
                cluster = clusters[i]
                if cluster.id in to_change:
                    continue
                if cluster.score < avg_score:
                    neighbor_clusters = self._get_neighbor_clusters(cluster)
                    for neighbor in reversed(neighbor_clusters):
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
                #centroids = [cluster.centroid for cluster in clusters]
                #clusters = self.assign_points_to_closest_centroid(points, centroids)
                score = self.calculate_score_unbalance(clusters)
                print(score)
                #if score < best_score:
                #    best_score = score
                #    best_clusters = deepcopy(clusters)
                #    print(f"Best score is {best_score}")
        return clusters# best_clusters

    def calculate_score_unbalance(self, clusters):
        max_score = max([cluster.score for cluster in clusters])
        min_score = min([cluster.score for cluster in clusters])
        return max_score / min_score

    def _merge_and_split(self, clusters, cluster1: Cluster, cluster2: Cluster):
        clusters.remove(cluster1)
        clusters.remove(cluster2)
        cluster1, cluster2 = cluster1.merge(cluster2).split()
        clusters.add(cluster1)
        clusters.add(cluster2)

    def _get_initial_split(self, points: List[Point]):
        cluster = Cluster(points)
        clusters = SortedList([cluster])
        for _ in range(self.n_clusters):
            biggest_cluster = clusters.pop(-1)
            c1, c2 = biggest_cluster.split()
            clusters.update([c1, c2])
        return clusters

    def _get_neighbor_clusters(self, cluster):
        neighbor_clusters = [neighbor.cluster for point in cluster.points for neighbor in point.neighbors]
        neighbor_clusters = SortedSet(neighbor_clusters)
        neighbor_clusters.remove(cluster)
        return neighbor_clusters
                    

        ##neighbors = list(
        ##    set(
        ##        indx
        ##        for simplex in delaunay.simplices
        ##        if i in simplex
        ##        for indx in simplex
        ##        if indx != i
        ##    )
        ##)
        #neighbor_clusters = [
        #    clusters[neighbor] for neighbor in neighbors if neighbor != -1
        #]
        #neighbor_scores = [cluster.score for cluster in neighbor_clusters]
        #return [neighbor_clusters[i] for i in np.argsort(neighbor_scores)[::-1]]
