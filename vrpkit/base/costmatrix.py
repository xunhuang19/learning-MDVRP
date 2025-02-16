from .demand import Task
from .facility import Asset
from .basics import Point

import random
from collections.abc import Iterable

import networkx as nx
import osmnx as ox
from scipy.spatial import distance


class CostMatrix:
    """A cost matrix which indexes the cost by node ids"""

    def __init__(self, node_ids=None, matrix=None):
        self.node_ids = list(node_ids) if node_ids is not None else []
        if matrix is not None:
            self._matrix = {
                node_i: {node_j: matrix[i][j] for j, node_j in enumerate(self.node_ids)}
                for i, node_i in enumerate(self.node_ids)
            }
        else:
            self._matrix = {}

    def __getitem__(self, key):
        return self._matrix.get(key, {})

    @property
    def matrix(self):
        return [[self._matrix[i][j] for j in self.node_ids] for i in self.node_ids]

    def cost(self, orig_id, dest_id):
        return self._matrix[orig_id][dest_id]

    def sub_matrix(self, sub_node_ids):
        sub_matrix = [[self._matrix[i][j] for j in sub_node_ids] for i in sub_node_ids]
        return CostMatrix(sub_node_ids, sub_matrix)

    def to_flatten_dict(self):
        return {(i, j): self._matrix[i][j] for i in self.node_ids for j in self.node_ids}

    def init_given_points(self, nodes: Iterable[Point]):
        unique_nodes_set = {n.node_id: n.coordinates for n in nodes}
        node_ids = list(unique_nodes_set.keys())
        node_coords = list(unique_nodes_set.values())
        matrix = distance.cdist(node_coords, node_coords, "euclidean")
        self.__init__(node_ids, matrix)

    def init_given_asset_and_task(self, asset: Asset, task: Task):
        all_points = [depot.location for depot in asset.depots.values()]
        for order in task.values():
            if order.loading_location:
                all_points.append(order.loading_location)
            elif order.unloading_location:
                all_points.append(order.unloading_location)
        self.init_given_points(all_points)


class NetworkMatrix(CostMatrix):
    def __init__(self, node_ids=None, matrix=None, graph=None, weight_label="length"):
        super().__init__(node_ids, matrix)
        self._graph = graph
        self.weight_label = weight_label

        # wrapping some useful functions from the osmnx package
        self.get_graph_from_polygon = self._save_network_graph(ox.graph_from_polygon)
        self.get_graph_from_address = self._save_network_graph(ox.graph_from_address)
        self.get_graph_from_place = self._save_network_graph(ox.graph_from_place)
        self.get_graph_from_bbox = self._save_network_graph(ox.graph_from_bbox)
        self.load_graph_from_graphml = self._save_network_graph(ox.load_graphml)

        self.save_graph_to_graphml = self._use_network_graph(ox.save_graphml)
        self.save_graph_to_geopackage = self._use_network_graph(ox.save_graph_geopackage)
        self.plot_route_on_graph = self._use_network_graph(ox.plot_graph_route)
        self.plot_graph = self._use_network_graph(ox.plot_graph)
        self.get_nearest_nodes = self._use_network_graph(ox.nearest_nodes)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph

    def sub_matrix(self, sub_node_ids):
        sub_matrix = [[self._matrix[i][j] for j in sub_node_ids] for i in sub_node_ids]
        return NetworkMatrix(sub_node_ids, sub_matrix, self.graph, self.weight_label)

    def shortest_path_cost(self, orig, dest, method="dijkstra"):
        """compute the cost of the shortest parth between two nodes on the network
        args:
            orig: the id of the origin node
            dest: the id of the destination node
            method: 'dijkstra' or 'bellman-ford'
        return
            path cost: a scalar measured by Network.weight_label
        """

        return nx.shortest_path_length(self._graph, orig, dest, weight=self.weight_label, method=method)

    def shortest_path(self, orig, dest, method="dijkstra"):
        """find the shortest parth between two nodes on the network
        args:
            orig: the id of the origin node
            dest: the id of the destination node
            method: 'dijkstra' or 'bellman-ford'
        return
            shortest_path: a list of edge ids.
        """

        return nx.shortest_path(self._graph, orig, dest, weight=self.weight_label, method=method)

    def update_link_cost(self):
        """update the cost stored in networkx.edges[edge_id]["weight_label"]
        This function should be defined by the user's requirements.
        """
        pass

    def random_pick_nodes(self, num=1):
        """randomly pick up a number of nodes
        args:
            num, the number of nodes
        return:
            node id or a list of node ids"""
        return random.choices(list(self._graph.nodes), k=num)

    def _use_network_graph(self, func):
        """a decorator for re-using osmnx functions but passing Network._graph to parameter-G"""

        def wrapper(*args, **kwargs):
            return func(self._graph, *args, **kwargs)

        return wrapper

    def _save_network_graph(self, func):
        """a decorator for re-using osmnx functions but saving returned graph to Network._graph"""

        def wrapper(*args, **kwargs):
            self._graph = func(*args, **kwargs)

        return wrapper
