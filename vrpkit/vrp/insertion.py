from vrpkit.base.solution import RoutePlan, Route
from vrpkit.base.objective import Objective, SpareLoad
from vrpkit.base.costmatrix import CostMatrix
from vrpkit.base.demand import Order, Task
from vrpkit.base.facility import Asset
from vrpkit.base.basics import ServicePoint
from vrpkit.base.constant import INF
from vrpkit.base.formulation import VRP
from vrpkit.base.solver import Solver
from vrpkit.vrp.two_opt import two_opt
from vrpkit.vrp.hga import optimal_vehicle_route_split

import math
import warnings
from collections.abc import Iterable
from itertools import product

import gurobipy as gp


def maximum_idle_time(orig: ServicePoint, dest: ServicePoint, travel_time):
    """calculate the maximum possible idle time from an origin node to a destination node"""
    # earliest arrival time from origin to destination
    arrival_t = orig.time_window.start + orig.service_duration + travel_time
    return max(dest.time_window.start - arrival_t, 0)


def maximum_service_delay(orig: ServicePoint, dest: ServicePoint, travel_time):
    """calculate the maximum possible service delay from an origin node to a destination node"""
    # latest arrival time from origin to destination
    arrival_t = orig.time_window.end + orig.service_duration + travel_time
    return max(arrival_t - dest.time_window.end, 0)


def euclidean_distance(orig: ServicePoint, dest: ServicePoint):
    x1, y1, x2, y2 = orig.long, orig.lat, dest.long, dest.lat
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def closeness(
        node1: ServicePoint, node2: ServicePoint,
        cost_matrix: CostMatrix = None, tts=None, objective: Objective = None,
        coef_tt=1, coef_dt=0, coef_it=0,
):
    """calculate the closeness (positive value) between two service points (orders)

    tts: a 2-element sequence, the first is tt_12 while the second is tt_21. tts
    will overwrite the cost_matrix. if no travel time information is provided, use
    euclidean distance.
    """
    # get coefficients
    if objective is not None:
        coef_tt = abs(objective.coefficients.get("Travel Time", 1))
        coef_dt = abs(objective.coefficients.get("Service Delay", 0))
        coef_it = abs(objective.coefficients.get("Idle Time", 0))

    # travel time
    if tts is None:
        if cost_matrix is not None:
            tt_12 = cost_matrix.cost(node1.node_id, node2.node_id)
            tt_21 = cost_matrix.cost(node2.node_id, node1.node_id)
        else:
            tt_12 = euclidean_distance(node1, node2)
            tt_21 = tt_12
    else:
        tt_12, tt_21 = tts[0], tts[1]
    tt = (tt_12 + tt_21) / 2

    # delay time
    if coef_dt == 0:
        dt = 0
    else:
        dt = (maximum_service_delay(node1, node2, tt_12) +
              maximum_service_delay(node2, node1, tt_21)) / 2

    # idle time
    if coef_it == 0:
        it = 0
    else:
        it = (maximum_idle_time(node1, node2, tt_12) +
              maximum_idle_time(node2, node1, tt_21)) / 2

    return coef_tt * tt + coef_dt * dt + coef_it * it


def simple_order_insertion(order: Order, solution: RoutePlan, vrp: VRP, count_depot=True, replace=False):
    """Insert one order into a delivery route plan to maximise the objective

    if order already exists in the route plan, the insertion profit is 0.
    if the spare capacity is not enough for insertion, the profit is -INF.

    """
    if solution.find(order.unloading_location) is not None:
        insert_profit = 0
        return solution, insert_profit

    # completely initiate route plan when not all vehicles are utilised
    unused_routes = RoutePlan()
    if len(solution) < len(vrp.asset.fleet):
        for veh_id in vrp.asset.fleet:
            if veh_id not in solution:
                depot_id = vrp.asset.halt[veh_id]
                depot_loc = vrp.asset.depots[depot_id].location
                unused_routes[veh_id] = Route([depot_loc, depot_loc])

    # find the nearest route
    closest_d = INF
    orig_closest_route = None
    closest_veh_id = None
    for veh_id, route in {**solution, **unused_routes}.items():
        centroid = route.centroid(count_depot)
        d = closeness(order.unloading_location, centroid, vrp.objective)
        spare_cap = SpareLoad.evaluate_route(veh_id, route, vrp.cost_matrix, vrp.asset, vrp.task)
        if d < closest_d and spare_cap >= order.volume:
            closest_d = d
            orig_closest_route = route
            closest_veh_id = veh_id

    # find the best insertion location in the closest route
    best_insert_location = None
    best_insert_profit = -INF
    if closest_veh_id is not None:
        orig_route_obj = vrp.evaluate_route(closest_veh_id, orig_closest_route)
        for i in range(1, len(orig_closest_route)):
            inserted_route = orig_closest_route.copy()
            inserted_route.insert(i, order.unloading_location)
            new_route_obj = vrp.evaluate_route(closest_veh_id, inserted_route)
            insert_profit = new_route_obj - orig_route_obj
            if insert_profit > best_insert_profit:
                best_insert_location = i
                best_insert_profit = insert_profit
    else:
        warnings.warn(f"Simple Order Insertion Failed: Order {order.id} cannot be inserted to {solution.display()}")

    inserted_solution = solution if replace else solution.copy()
    if best_insert_location is not None:
        inserted_closest_route = orig_closest_route.insert(best_insert_location, order.unloading_location)
        inserted_solution[closest_veh_id].insert(inserted_closest_route)

    return inserted_solution, best_insert_profit


def simple_bundle_insertion(orders: Iterable[Order], solution: RoutePlan, vrp: VRP, count_depot=True, replace=False):
    """Insert an order set (one by one) into a route plan to maximise the objective"""
    inserted_solution = solution if replace else solution.copy()
    total_insert_profit = 0
    for order in orders:
        _, profit = simple_order_insertion(order, inserted_solution, vrp, count_depot, True)
        total_insert_profit += profit
    return inserted_solution, total_insert_profit


class NearestInsertionHeuristic(Solver):

    def __init__(self, vrp: VRP = None):
        super().__init__(vrp)
        self.depot_assignment_method = "exact"
        self.do_post_2opt = False
        self.post_2opt_iter_n = 10
        self.do_post_opt_split = True

    def run(self, *args, **kwargs):
        super(NearestInsertionHeuristic, self).run(*args, **kwargs)
        if self.depot_assignment_method == "exact":
            depot_orders = exact_nearest_depot_assignment(self.vrp.task, self.vrp.asset, self.vrp.cost_matrix)
        else:
            depot_orders = approx_nearest_depot_assignment(self.vrp.task, self.vrp.asset, self.vrp.cost_matrix)

        route_plan = RoutePlan()
        for depot_id, assigned_order_ids in depot_orders.items():
            sub_asset = self.vrp.asset.subset(depot_id)
            sub_task = self.vrp.task.subset(assigned_order_ids)
            sub_route_plan = single_depot_simple_nearest_insertion(sub_task, sub_asset, self.vrp.objective)

            if self.do_post_opt_split:
                routes = sub_route_plan.values()
                orders = [self.vrp.task[unload_loc.order_id] for route in routes for unload_loc in route[1:-1]]
                sub_route_plan = optimal_vehicle_route_split(self.vrp.asset.depots[depot_id],
                                                             self.vrp.asset,
                                                             orders,
                                                             self.vrp.cost_matrix)

            route_plan += sub_route_plan

        if self.do_post_2opt:
            two_opt(route_plan, self.vrp, self.post_2opt_iter_n)

        # if len(route_plan) == 1:

        self.best_solution = route_plan
        self._vrp.solution = self.best_solution


def single_depot_simple_nearest_insertion(task: Task, asset: Asset, objective: Objective):
    """assign orders to a fleet under the same depot. I.e., solve a vrp by the simple nearest insertion"""

    # only proceed for one depot or the first depot
    if len(asset.depots) == 1:
        depot = asset.depots.values_in_list()[0]
        vehicles = asset.fleet.values_in_list()
    elif len(asset.depots) > 1:
        depot = asset.depots.values_in_list()[0]
        vehicles = asset.depot_vehicles(depot.id)[depot.id]
    else:
        raise ValueError("Asset can not be empty for single-depot simple nearest insertion")

    # initialized route plan
    route_plan = RoutePlan()
    task = task.copy()

    # create vehicle route by iteratively inserting the order closest to current route
    for veh in vehicles:
        if task.is_empty():
            break

        spare_cap = veh.capacity
        no_cap = False
        route = Route([depot.location, depot.location])
        while not task.is_empty() and not no_cap:
            nearest_order_id = None
            nearest_order_dist = INF
            for order_id, order in task.items():
                if spare_cap >= order.volume:
                    centroid = route.centroid(count_depot=True)
                    dist = closeness(centroid, order.unloading_location, objective=objective)
                    if nearest_order_dist > dist:
                        nearest_order_id = order_id
                        nearest_order_dist = dist
                else:
                    no_cap = True
                    break

            if nearest_order_id is not None:
                nearest_order = task[nearest_order_id]
                route.insert(-1, nearest_order.unloading_location)
                spare_cap -= nearest_order.volume
                task.pop(nearest_order_id)

        route_plan[veh.id] = route


    if not task.is_empty():
        warnings.warn(f"orders {task.keys_in_list()} are not inserted successfully due to insufficient capacity")
        last_veh = vehicles[-1]
        for order in task.values():
            route_plan[last_veh.id].insert(-1, order.unloading_location)

    return route_plan


def exact_nearest_depot_assignment(task: Task, asset: Asset, cost_matrix: CostMatrix):
    order_ids = task.keys_in_list()
    depot_ids = asset.depots.keys_in_list()
    depot_vehicles = asset.depot_vehicles()
    capacities = {j: sum([v.capacity for v in vehicles]) for j, vehicles in depot_vehicles.items()}
    volumes = task.get_attributes("volume")
    dist_matrix = {}
    for order_id, depot_id in product(order_ids, depot_ids):
        order, depot = task[order_id], asset.depots[depot_id]
        dist_matrix[order_id, depot_id] = closeness(order.unloading_location, depot.location, cost_matrix)

    # decide the assignment by formulating an integer programming
    model = gp.Model("order assignment")
    x = model.addVars(order_ids, depot_ids, name="order_alloc", vtype=gp.GRB.BINARY)
    model.setObjective(sum(dist_matrix[i, j] * x[i, j] for i, j in product(order_ids, depot_ids)), gp.GRB.MINIMIZE)
    # one order must be assigned exactly once
    model.addConstrs(x.sum(i, "*") == 1 for i in order_ids)
    # capacity constraint
    model.addConstrs((sum(volumes[i] * x[i, j] for i in order_ids) <= capacities[j] for j in depot_ids))
    model.optimize()
    # decode the results
    assignment = {depot_id: [] for depot_id in depot_ids}
    for i, j in product(order_ids, depot_ids):
        if x[i, j].X == 1:
            assignment[j].append(i)

    return assignment


def approx_nearest_depot_assignment(task: Task, asset: Asset, cost_matrix: CostMatrix):
    order_ids, depot_ids = task.keys_in_list(), asset.depots.keys_in_list()
    dist_matrix = {}
    for order_id, depot_id in product(order_ids, depot_ids):
        order, depot = task[order_id], asset.depots[depot_id]
        if dist_matrix.get(order_id) is None:
            dist_matrix[order_id] = {}
        dist_matrix[order_id][depot_id] = closeness(order.unloading_location, depot.location, cost_matrix)

    # sort the orders by its minimum distance to all depots
    order_min_dist = {i: min(dist_matrix[i].values()) for i in order_ids}
    order_prioritized = sorted(order_min_dist, key=order_min_dist.get)

    depot_vehicles = asset.depot_vehicles()
    spare_cap = {depot_id: sum([v.capacity for v in vehicles]) for depot_id, vehicles in depot_vehicles.items()}
    assignment = {depot_id: [] for depot_id in depot_ids}
    for order_id in order_prioritized:
        assigned = False
        order_vol = task[order_id].volume

        # sort depots from nearest to farthest
        depot_dist = dist_matrix[order_id]
        depots_prioritized = sorted(depot_dist, key=depot_dist.get)

        for depot_id in depots_prioritized:
            if spare_cap[depot_id] >= order_vol:
                assignment[depot_id].append(order_id)
                spare_cap[depot_id] -= order_vol
                assigned = True
                break

        if not assigned:
            depot_id = depots_prioritized[0]
            assignment[depot_id].append(order_id)
            spare_cap[depot_id] -= order_vol
            warnings.warn(f"Depot Order Assignment Warning: capacity constraint is violate for depot {depot_id}")

    return assignment
