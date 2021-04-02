import pytest
from score_clustering import Point, sort_points

class TestPoints:
    def test__init(self):
        p = Point(3,4,5)
        assert p.x == 3
        assert p.y == 4
        assert p.score == 5

    def test__point_order(self):
        p = Point(1,2)
        q = Point(2,3)
        assert p < q
        p = Point(1,2)
        q = Point(1,2)
        assert p == q
        p = Point(2,3)
        q = Point(1,3)
        assert p > q

    def test__create_point_set(self):
        points_list = [Point(1,2), Point(4,5), Point(-2,3), Point(1,5)]
        points = sort_points(points_list)
        for point in points_list:
            assert point in points
        assert points[0] == Point(1,2)
        assert points[1] == Point(-2,3)
        assert points[2] == Point(1,5)
        assert points[3] == Point(4,5)


