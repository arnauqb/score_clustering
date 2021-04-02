import pytest
import numpy as np
import random

# fix random seed for tests
random.seed(0)
np.random.seed(0)
from sortedcontainers import SortedSet

from score_clustering import sort_points, Cluster, Point, get_cluster_split, merge


class TestClusterClass:
    def test_cluster(self):
        points = list(map(lambda x: Point(*x), [[1, 2, 10], [5, 1, 2], [8, 1, 3]]))
        cluster = Cluster(points)
        assert len(cluster) == 3
        assert cluster[0] == Point(1, 2, 10)
        assert cluster[1] == Point(5, 1, 2)
        assert cluster[2] == Point(8, 1, 3)
        assert (np.isclose(cluster.centroid, [14 / 3, 4 / 3])).all()
        assert cluster.score == 15
        points2 = list(map(lambda x: Point(*x), [[1, 2, 10], [5, 1, 2], [8, 1, 3]]))
        cluster2 = Cluster(points2)
        assert cluster == cluster2
        points3 = list(map(lambda x: Point(*x), [[1, 2, 10], [8, 1, 3]]))
        cluster3 = Cluster(points3)
        assert cluster != cluster3
        assert cluster3.score == 13
        assert (np.isclose(cluster3.centroid, [9 / 2, 3 / 2])).all()

    def test__split_cluster(self):
        points = list(
            map(lambda x: Point(*x), [[1, 2, 10], [3, 4, 5], [5, 6, 2], [8, 9, 3]])
        )
        cluster = Cluster(points)
        cluster1, cluster2 = cluster.split([0,0], [0,1])
        assert cluster.score == 20
        assert len(cluster1) == 1
        assert cluster1[0] == Point(1, 2, 10)
        assert cluster1.score == 10
        assert len(cluster2) == 3
        assert cluster2.score == 10
        assert cluster2[0] == Point(3, 4, 5)
        assert cluster2[1] == Point(5, 6, 2)
        assert cluster2[2] == Point(8, 9, 3)
        assert len(cluster.subclusters) == 2
        assert cluster.subclusters[0] == cluster1
        assert cluster.subclusters[1] == cluster2
        # test split again doesnt add new subclusters
        cluster.split([0,0], [0,1])
        assert len(cluster.subclusters) == 2
        assert cluster.subclusters[0] == cluster1
        assert cluster.subclusters[1] == cluster2

    def test__merge_cluster(self):
        points = list(map(lambda x: Point(*x), [[1, 2, 10], [5, 6, 2]]))
        cluster1 = Cluster(points)
        points = list(map(lambda x: Point(*x), [[3, 4, 5], [8, 9, 3]]))
        cluster2 = Cluster(points)
        cluster = merge(cluster1, cluster2)
        assert len(cluster) == 4
        assert cluster[0] == Point(1, 2, 10)
        assert cluster[1] == Point(5, 6, 2)
        assert cluster[2] == Point(3, 4, 5)
        assert cluster[3] == Point(8, 9, 3)


class TestClustering:
    def test__all_points_same_score(self):
        points = list(map(lambda x: Point(*x), 100 * np.random.random((100, 3))))
        for point in points:
            point.score = 10
        total_score = sum(point.score for point in points)
        n_clusters = 10
        avg_score = total_score / 10
        clusters = get_cluster_split(points, n_clusters, niters=20)
        assert len(clusters) == 10
        for cluster in clusters:
            assert np.isclose(cluster.score, avg_score, rtol=0.1)

    def test__all_points_similar_score(self):
        points = list(map(lambda x: Point(*x), 100 * np.random.random((100, 3))))
        for point in points:
            point.score = 10 + random.random()
        total_score = sum(point.score for point in points)
        n_clusters = 10
        avg_score = total_score / 10
        clusters = get_cluster_split(points, n_clusters, niters=10)
        assert len(clusters) == 10
        for cluster in clusters:
            assert np.isclose(cluster.score, avg_score, rtol=0.2)
