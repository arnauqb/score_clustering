from sortedcontainers import SortedSet

from score_clustering import sort_points, Cluster, Point

class TestClusterClass:
    def test_cluster(self):
        points = map(lambda x: Point(*x), [[1, 2, 10], [5, 1, 2], [8, 1, 3]])
        cluster = Cluster(points)
        assert len(cluster) == 3
        assert cluster[0] == Point(5,1,2)
        assert cluster[1] == Point(8,1,3)
        assert cluster[2] == Point(1,2,10)
        points2 = map(lambda x: Point(*x), [[1, 2, 10], [5, 1, 2], [8, 1, 3]])
        cluster2 = Cluster(points2)
        assert cluster == cluster2
        points3 = map(lambda x: Point(*x), [[1, 2, 10], [8, 1, 3]])
        cluster3 = Cluster(points3)
        assert cluster != cluster3

    def test__split_cluster(self):
        points = map(lambda x: Point(*x), [[1, 2, 10], [3, 4, 5], [5, 6, 2], [8, 9, 3]])
        cluster = Cluster(points)
        cluster1, cluster2 = cluster.split()
        assert len(cluster1) == 1
        assert cluster1[0] == Point(1, 2, 10)
        assert len(cluster2) == 3
        assert cluster2[0] == Point(3, 4, 5)
        assert cluster2[1] == Point(5, 6, 2)
        assert cluster2[2] == Point(8, 9, 3)
        assert len(cluster.subclusters) == 2
        assert cluster.subclusters[0] == cluster1
        assert cluster.subclusters[1] == cluster2
        # test split again doesnt add new subclusters
        cluster.split()
        assert len(cluster.subclusters) == 2
        assert cluster.subclusters[0] == cluster1
        assert cluster.subclusters[1] == cluster2

