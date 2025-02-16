from .solution import RoutePlan, Route
from .facility import Asset
from .demand import Task, UnloadingPoint, LoadingPoint
from .costmatrix import CostMatrix

from collections.abc import Sequence


class Metric:
    name = "metric"

    @classmethod
    def evaluate(cls, solution: RoutePlan, cost_matrix: CostMatrix, asset: Asset, task: Task):
        """calculate the value of the metric for a solution"""
        value = 0
        for vehicle_id in solution:
            vehicle_route = solution[vehicle_id]
            value += cls.evaluate_route(vehicle_id, vehicle_route, cost_matrix, asset, task)
        return value

    @staticmethod
    def evaluate_route(vehicle_id, route: Route, cost_matrix: CostMatrix, asset: Asset, task: Task):
        """calculate the value of the metric for a route of a solution"""
        route_value = 0
        return route_value


class TravelTime(Metric):
    name = "Travel Time"

    @staticmethod
    def evaluate_route(vehicle_id, route: Route, cost_matrix: CostMatrix, asset: Asset, task: Task):
        tt = 0
        for pre_node, next_node in zip(route[:-1], route[1:]):
            tt += cost_matrix.cost(pre_node.node_id, next_node.node_id)
        return tt


class ServiceDelay(Metric):
    name = "Service Delay"

    @staticmethod
    def evaluate_route(vehicle_id, route: Route, cost_matrix: CostMatrix, asset: Asset, task: Task):
        delay = 0
        if len(route) != 0:
            start_time = route[0].time_window.start
            time_spent = start_time + route[0].service_duration
            for i in range(1, len(route)):
                pre_node, next_node = route[i - 1], route[i]
                time_spent += cost_matrix.cost(pre_node.node_id, next_node.node_id)
                tw_end = next_node.time_window.end
                delay += max(time_spent - tw_end, 0)
                time_spent += next_node.service_duration
        return delay


class IdleTime(Metric):
    name = "Idle Time"

    @staticmethod
    def evaluate_route(vehicle_id, route: Route, cost_matrix: CostMatrix, asset: Asset, task: Task):
        idle_time = 0
        if len(route) != 0:
            start_time = route[0].time_window.start
            time_spent = start_time + route[0].service_duration
            for i in range(1, len(route)):
                pre_node, next_node = route[i - 1], route[i]
                time_spent += cost_matrix.cost(pre_node.node_id, next_node.node_id)
                tw_start = next_node.time_window.start
                idle_time += max(tw_start - time_spent, 0)
                time_spent += next_node.service_duration
        return idle_time


class GrossRevenue(Metric):
    name = "Gross Revenue"

    @staticmethod
    def evaluate_route(vehicle_id, route: Route, cost_matrix: CostMatrix, asset: Asset, task: Task):
        gross_revenue = 0
        order_ids = set([point.order_id for point in route if isinstance(point, (UnloadingPoint, LoadingPoint))])
        gross_revenue += sum([task[order_id].value for order_id in order_ids])
        return gross_revenue


class OverLoad(Metric):
    name = "Over Load"

    @staticmethod
    def evaluate_route(vehicle_id, route: Route, cost_matrix: CostMatrix, asset: Asset, task: Task):
        vehicle = asset.fleet[vehicle_id]
        capacity = vehicle.capacity
        order_ids = set([point.order_id for point in route if isinstance(point, (UnloadingPoint, LoadingPoint))])
        volume = sum([task[order_id].volume for order_id in order_ids])
        overload = 0 if volume <= capacity else volume - capacity
        return overload


class SpareLoad(Metric):
    name = "Spare Load"

    @staticmethod
    def evaluate_route(vehicle_id, route: Route, cost_matrix: CostMatrix, asset: Asset, task: Task):
        vehicle = asset.fleet[vehicle_id]
        capacity = vehicle.capacity
        order_ids = set([point.order_id for point in route if isinstance(point, (UnloadingPoint, LoadingPoint))])
        volume = sum([task[order_id].volume for order_id in order_ids])
        spareload = 0 if volume >= capacity else capacity - volume
        return spareload


class VehicleNumber(Metric):
    name = "Vehicle Number"

    @staticmethod
    def evaluate_route(vehicle_id, route: Route, cost_matrix: CostMatrix, asset: Asset, task: Task):
        veh_used = 0
        for point in route:
            if isinstance(point, (UnloadingPoint, LoadingPoint)):
                veh_used = 1
        return veh_used


class Objective:
    """objective, a component of VRP form, defines the calculation and importance of metrics"""

    def __init__(self, metrics: Sequence[Metric] = None, coefficients: Sequence[float] = None):
        self.metrics: dict[str, Metric] = {metric.name: metric for metric in metrics} if metrics is not None else {}
        if coefficients is not None:
            if len(self.metrics) != len(coefficients):
                raise ValueError("number of coefficients must equal to the number of metrics")
            else:
                self.coefficients = {name: coefficients[i] for i, name in enumerate(self.metrics)}
        else:
            self.coefficients = {name: 1 for name in self.metrics}
        self.values = {name: 0 for name in self.metrics}

    def evaluate(self, solution: RoutePlan, cost_matrix: CostMatrix, asset: Asset, task: Task):
        obj_value = 0
        for metric_name in self.metrics:
            value = self.metrics[metric_name].evaluate(solution, cost_matrix, asset, task)
            self.values[metric_name] = value
            obj_value += self.coefficients[metric_name] * value
        return obj_value

    def evaluate_route(self, vehicle_id, route: Route, cost_matrix: CostMatrix, asset: Asset, task: Task):
        obj_value = 0
        for metric_name in self.metrics:
            value = self.metrics[metric_name].evaluate_route(vehicle_id, route, cost_matrix, asset, task)
            self.values[metric_name] = value
            obj_value += self.coefficients[metric_name] * value
        return obj_value

    def add_metric(self, metric, coefficient):
        self.metrics[metric.name] = metric
        self.coefficients[metric.name] = coefficient
        self.values[metric.name] = 0
