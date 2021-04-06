import pytest
import numpy as np
import random

# fix random seed for tests
random.seed(0)
np.random.seed(0)
from sortedcontainers import SortedSet

from score_clustering import Cluster, Point, ScoreClustering


class TestClusterClass:
    def test_cluster(self):
        points = list(map(lambda x: Point(*x), [[1, 2, 10], [5, 1, 2], [8, 1, 3]]))
        cluster = Cluster(points)
        assert len(cluster) == 3
        assert cluster[0] == Point(1, 2, 10)
        assert cluster[1] == Point(5, 1, 2)
        assert cluster[2] == Point(8, 1, 3)
        assert (np.isclose([cluster.centroid.x, cluster.centroid.y], [14 / 3, 4 / 3])).all()
        assert cluster.score == 15
        points2 = list(map(lambda x: Point(*x), [[1, 2, 10], [5, 1, 2], [8, 1, 3]]))
        cluster2 = Cluster(points2)
        assert cluster == cluster2
        points3 = list(map(lambda x: Point(*x), [[1, 2, 10], [8, 1, 3]]))
        cluster3 = Cluster(points3)
        assert cluster != cluster3
        assert cluster3.score == 13
        assert (np.isclose([cluster3.centroid.x, cluster3.centroid.y], [9 / 2, 3 / 2])).all()


class TestClustering:
    def test__all_points_same_score(self):
        points = list(map(lambda x: Point(*x), 100 * np.random.random((100, 3))))
        for point in points:
            point.score = 10
        total_score = sum(point.score for point in points)
        n_clusters = 10
        avg_score = total_score / 10
        score_clustering = ScoreClustering(n_clusters)
        clusters = score_clustering.fit(points, niters=10)
        assert len(clusters) == 10
        for cluster in clusters:
            assert np.isclose(cluster.score, avg_score, rtol=0.3)

    def test__all_points_similar_score(self):
        points = list(map(lambda x: Point(*x), 100 * np.random.random((100, 3))))
        for point in points:
            point.score = 10 + random.random()
        total_score = sum(point.score for point in points)
        n_clusters = 10
        avg_score = total_score / 10
        score_clustering = ScoreClustering(n_clusters)
        clusters = score_clustering.fit(points, niters=10)
        assert len(clusters) == 10
        for cluster in clusters:
            assert np.isclose(cluster.score, avg_score, rtol=0.3)
