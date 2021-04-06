import pytest
import numpy as np

from score_clustering import Point, score_vector, calculate_centroid_update


class TestPoints:
    def test__init(self):
        p = Point(3, 4, 5, "test")
        assert p.x == 3
        assert p.y == 4
        assert p.score == 5
        assert p.name == "test"

    def test__score_vector(self):
        avg_score = 2
        p = Point(1, 2, 10)
        q = Point(2, 3, 5)
        assert np.allclose(
            score_vector(p, q, avg_score), -np.log(7 / 2) * np.array([1, 1]), rtol=0.05
        )
        p = Point(1, 2, 5)
        q = Point(2, 3, 10)
        assert np.allclose(
            score_vector(p, q, avg_score), np.log(7 / 2) * np.array([1, 1]), rtol=0.05
        )
        p = Point(6, 2, 5)
        q = Point(1, 3, 10)
        assert np.allclose(
            score_vector(p, q, avg_score), np.log(7 / 2) * np.array([-5, 1]), rtol=0.05
        )

    def test__calculate_centroid_update(self):
        points = [
            Point(0, 0, 1),
            Point(10, 0, 2),
            Point(0, 10, 3),
            Point(10, 10, 4),
        ]
        update_vector = calculate_centroid_update(Point(5,5,3), points, 2.5)
        assert np.isclose(update_vector[0], 2.939, rtol=1e-2)
        assert np.isclose(update_vector[1], 6.304, rtol=1e-2)

