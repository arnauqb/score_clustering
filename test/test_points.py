import pytest
import numpy as np

from score_clustering import Point, get_intersection_time, sort_points


class TestPoints:
    def test__init(self):
        p = Point(3, 4, 5, "test")
        assert p.x == 3
        assert p.y == 4
        assert p.score == 5
        assert p.name == "test"

    def test__sort_points(self):
        A = Point(2.46, 4.83, 0)
        B = Point(2, 2, 0)
        C = Point(5, 2.49, 0)
        D = Point(5, 4, 0)
        E = Point(4.18, 7.15, 0)
        points = np.array([A, B, C, D, E])

        # Delaunay vertices
        F = np.array([0.3, 2.57, 0])
        G = np.array([9, 6, 0])

        # calculated with Geogebra
        assert np.isclose(get_intersection_time(F, G, A), 2.84, rtol=0.01)
        assert np.isclose(get_intersection_time(F, G, B), 1.37, rtol=0.01)
        assert np.isclose(get_intersection_time(F, G, C), 4.34, rtol=0.01)
        assert np.isclose(get_intersection_time(F, G, D), 4.9, rtol=0.01)
        assert np.isclose(get_intersection_time(F, G, E), 5.29, rtol=0.01)

        sorted_points = sort_points(points, F, G)
        assert sorted_points[0] == B
        assert sorted_points[1] == A
        assert sorted_points[2] == C
        assert sorted_points[3] == D
        assert sorted_points[4] == E
