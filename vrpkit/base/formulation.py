from .facility import Asset
from .demand import Task
from .solution import RoutePlan
from .objective import Objective
from .costmatrix import CostMatrix
from .solution import Route


class VRP:

    def __init__(self, asset: Asset = None, task: Task = None, cost_matrix: CostMatrix = None,
                 objective: Objective = None):
        self.asset = asset if asset is not None else Asset()
        self.task = task if task is not None else Task()
        self.cost_matrix = cost_matrix if cost_matrix is not None else CostMatrix()
        self.objective = objective if objective is not None else Objective()
        self.solution = RoutePlan()
        self.validate_data()

    def __add__(self, other):
        inst = self.__class__(self.asset + other.asset,
                              self.task + other.task,
                              self.cost_matrix,
                              self.objective)
        return inst

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def validate_data(self):
        # todo： check whether cost matrix is complete and whether objective is non-empty
        pass

    def is_empty(self):
        return self.asset.is_empty() and self.task.is_empty()

    def copy(self):
        """self-define copy"""
        return self.__class__(self.asset.copy(), self.task.copy(), self.cost_matrix, self.objective)

    def visualize(self, ax, solution: RoutePlan = None,
                  c_path="blue", arrow_width=0.02, l_depo="Depot", c_depo="orange",
                  l_cust="Customer", c_cust_ulp="green", c_cust_lp="red", legend=True, annotate=True):
        """visualize route plan through matplotlib"""

        solution = self.solution if solution is None else solution

        # visualize routes
        for vehicle_id in solution:
            route = solution[vehicle_id]
            for pre_n, next_n in zip(route[:-1], route[1:]):
                ax.arrow(
                    pre_n.long,
                    pre_n.lat,
                    next_n.long - pre_n.long,
                    next_n.lat - pre_n.lat,
                    color=c_path,
                    head_width=arrow_width,
                )

        # plot depots
        depot_loc_coords = self.asset.depots.location_coords()
        ax.scatter([long for long, _ in depot_loc_coords.values()],
                   [lat for _, lat in depot_loc_coords.values()],
                   marker="^",
                   color=c_depo,
                   label=l_depo)
        if annotate:
            for depot_id, (long, lat) in depot_loc_coords.items():
                ax.annotate(f"{depot_id}", (long, lat))

        # plot unloading points
        order_unload_loc_coords = self.task.unloading_location_coords()
        if len(order_unload_loc_coords):
            ax.scatter([long for long, _ in order_unload_loc_coords.values()],
                       [lat for _, lat in order_unload_loc_coords.values()],
                       color=c_cust_ulp,
                       label=f"{l_cust} Unload")
        if annotate:
            for order_id, (long, lat) in order_unload_loc_coords.items():
                ax.annotate(f"{order_id}", (long, lat))

        # plot loading points
        order_load_loc_coords = self.task.loading_location_coords()
        if len(order_load_loc_coords):
            ax.scatter([long for long, _ in order_load_loc_coords.values()],
                       [lat for _, lat in order_load_loc_coords.values()],
                       color=c_cust_lp,
                       label=f"{l_cust} Load")
        if annotate:
            for order_id, (long, lat) in order_load_loc_coords.items():
                ax.annotate(f"{order_id}", (long, lat))

        if legend:
            ax.legend()

        return ax

    def evaluate(self, solution: RoutePlan = None):
        """get values of the objective and metrics"""
        if solution is None:
            solution = self.solution
        obj_value = self.objective.evaluate(solution, self.cost_matrix, self.asset, self.task)

        return obj_value, self.objective.values

    def evaluate_route(self, vehicle_id, route: Route):
        """compute the objective and metric values for a single vehicle route"""

        obj_value = self.objective.evaluate_route(vehicle_id, route, self.cost_matrix, self.asset, self.task)
        return obj_value, self.objective.values
