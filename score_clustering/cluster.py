from typing import List
import numpy as np
from scipy.spatial import distance_matrix
from itertools import count
import networkx as nx

from .point import Point 


class Cluster:
    """
    Represents a cluster of points.
    """

    _id = count()

    def __init__(self, points: List[Point]):
        self.id = next(self._id)
        self._graph = nx.Graph()
        self._graph.add_nodes_from(points)
        self.points = set(points)
        for point in self.points:
            for neighbor in point.neighbors:
                if neighbor in self.points:
                    self._graph.add_edge(point, neighbor)
        #self.connected_components = [self._graph.subgraph(c).copy() for c in nx.connected_components(self._graph)]
        #self.articulation_points = [nx.articulation_points(c) for c in self.connected_components]
        self.articulation_points = nx.articulation_points(self._graph)
        for point in points:
            point.cluster = self

    def is_articulation_point(self, point):
        return point in self.articulation_points
        #for i, c in enumerate(self.connected_components):
        #    if point in c:
        #        return point in self.articulation_points[i]
        #connected_component = self._graph.subgraph(nx.node_connected_component(self._graph, point))
        #ret = point in nx.articulation_points(connected_component)
        #return ret

    def add(self, point):
        self._graph.add_node(point)
        for neighbor in point.neighbors:
            if neighbor == point:
                continue
            if neighbor in self.points:
                self._graph.add_edge(point, neighbor)
        self.points.add(point)
        point.cluster = self
        self.articulation_points = nx.articulation_points(self._graph)
        #self.connected_components = [self._graph.subgraph(c).copy() for c in nx.connected_components(self._graph)]
        #self.articulation_points = [nx.articulation_points(c) for c in self.connected_components]

    def remove(self, point):
        self._graph.remove_node(point)
        self.points.remove(point)
        point.cluster = None
        self.articulation_points = nx.articulation_points(self._graph)
        #self.connected_components = [self._graph.subgraph(c).copy() for c in nx.connected_components(self._graph)]
        #self.articulation_points = [nx.articulation_points(c) for c in self.connected_components]

    def __getitem__(self, index):
        return self.points[index]

    def __len__(self):
        return len(self.points)

    def __hash__(self):
        return hash(self.id)

    @property
    def score(self):
        return sum(point.score for point in self.points)
