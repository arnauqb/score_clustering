from typing import List
from sortedcontainers import SortedSet


class Point:
    """
    A class that represents a point.
    An order between points is defined such that
        p < q if
            p.y < q.y
            or
            p.y == q.y and p.x < q.x
    """

    def __init__(self, x: float, y: float, score: float = 0):
        self.x = x
        self.y = y
        self.score = score

    def __lt__(self, other):
        if self.y < other.y:
            return True
        elif self.y == other.y:
            return self.x < other.x
        else:
            return False

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)


def sort_points(points: List[Point]):
    return SortedSet(points)
