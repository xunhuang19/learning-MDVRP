from .basics import ServicePoint, DictSet
from .demand import UnloadingPoint, LoadingPoint, Order
from .facility import DepotServicePoint, Depot

from collections import UserList
from collections.abc import Iterable
import warnings


class Route(UserList):
    "a list of visiting points"

    def __init__(self, points: list[ServicePoint] = None):
        super(Route, self).__init__(points)

    def centroid(self, count_depot=True):
        """calculate the centroid of a route. the centroid is a virtual service point,
        of which attributes are the average values of all nodes. the inclusion of depots
        is optional"""
        centroid = ServicePoint(long=0, lat=0, tw_start=0, tw_end=0, duration=0)
        counted_points = self if count_depot else [p for p in self if not isinstance(p, DepotServicePoint)]
        counted_n = len(counted_points)
        for p in counted_points:
            centroid.long += p.long / counted_n
            centroid.lat += p.lat / counted_n
            centroid.time_window.start += p.time_window.start / counted_n
            centroid.time_window.end += p.time_window.end / counted_n
            centroid.service_duration += p.service_duration / counted_n
        if counted_n == 0:
            warnings.warn(
                f"Route Centroid Calculation Warning: empty route when{'' if count_depot else ' not'} counting depots"
            )
        return centroid

    def display(self, print2console=True):
        route_str = ""
        route_len = len(self)
        for i, point in enumerate(self):
            if isinstance(point, DepotServicePoint):
                route_str += str(point.depot_id)
            if isinstance(point, (UnloadingPoint, LoadingPoint)):
                route_str += str(point.order_id)
            if i != (route_len - 1):
                route_str += "->"
        if print2console:
            print(route_str)
        return route_str


def assemble_delivery_route(depot: Depot, orders: Iterable[Order]):
    return Route([depot.location] + [i.unloading_location for i in orders] + [depot.location])


class RoutePlan(DictSet):
    "a dict-like dataset of vehicle and route pairs, {vehicle_id: route}"

    def __init__(self, data: dict[[], Route] = None, /, **kwargs):
        super(RoutePlan, self).__init__(data, **kwargs)

    def copy(self):
        return self.__class__({veh_id: route.copy() for veh_id, route in self.items()})

    def find(self, point):
        for veh_id, route in self.items():
            if point in route:
                return veh_id, route.index(point)

    def replace(self, old_item: ServicePoint, new_item: ServicePoint):
        replaced = False
        for route in self.data.values():
            if old_item in route:
                idx = route.index(old_item)
                route[idx] = new_item
                replaced = True
        if not replaced:
            raise ValueError(f"Point {old_item} is not in this route plan {self}")

    def remove(self, item: ServicePoint):
        removed = False
        for route in self.data.values():
            for i, point in enumerate(route):
                if point is item:
                    route.pop(i)
                    removed = True
        if not removed:
            raise ValueError(f"Point {item} is not in this route plan {self}")

    def display(self, print2console=True):
        plan_str = "\n"
        for veh_id, route in self.items():
            route_str = ""
            route_len = len(route)
            not_empty = False  # an empty route is a route not visiting any (un)loading points
            for i, point in enumerate(route):
                if isinstance(point, DepotServicePoint):
                    route_str += str(point.depot_id)
                if isinstance(point, (UnloadingPoint, LoadingPoint)):
                    route_str += str(point.order_id)
                    not_empty = True
                if i != (route_len - 1):
                    route_str += "->"
            if not_empty:
                plan_str += f"For vehicle {veh_id}, the route is {route_str} \n"
        if print2console:
            print(plan_str)
        return plan_str
