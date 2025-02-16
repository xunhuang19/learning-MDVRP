from ..base.solver import Solver
from ..base.formulation import VRP
from ..base.solution import Route, RoutePlan
from ..base.utils import powerset

from itertools import product

import gurobipy as gp


class GurobiSolver(Solver):
    """An exact solver for single-depot capacitated vehicle routing problem"""

    def __init__(self, vrp=None):
        super(GurobiSolver, self).__init__(vrp)
        self.depots = []
        self.orders = []
        self.vertices = []
        self.order_dmds = {}
        self.vehicles = []
        self.depot_vehs = {}
        self.veh_caps = {}
        self.distance_matrix = {}
        self.model = None
        self.flow_vars = None
        self.encode()

    def encode(self):
        depot2loc = self.vrp.asset.depots.location_ids()
        order2loc = self.vrp.task.unloading_location_ids()
        vertex2loc_id = {**depot2loc, **order2loc}

        self.depots = self.vrp.asset.depots.keys_in_list()
        self.orders = self.vrp.task.keys_in_list()
        self.vertices = self.depots + self.orders
        self.vehicles = self.vrp.asset.fleet.keys_in_list()
        self.depot_vehs = self.vrp.asset.depot_vehicle_ids()
        self.order_dmds = self.vrp.task.get_attributes("volume")
        self.veh_caps = self.vrp.asset.fleet.get_attributes("capacity")
        self.distance_matrix = {(i, j): self.vrp.cost_matrix.cost(vertex2loc_id[i], vertex2loc_id[j])
                                for i, j in product(self.vertices, self.vertices)}

    def initialize(self, vrp: VRP = None, *args, **kwargs):
        super(GurobiSolver, self).initialize(vrp)

        # create the gurobi optimisation model
        model = gp.Model("c-vrp")
        # create decision vars - vehicle flows
        x = model.addVars(self.vertices, self.vertices, self.vehicles, name="vehicle_flow", vtype=gp.GRB.BINARY)
        # set the objective
        model.setObjective(sum(self.distance_matrix[i, j] * x[i, j, v] for i, j, v in
                               product(self.vertices, self.vertices, self.vehicles)), gp.GRB.MINIMIZE)
        # add constraints - order is only visited once
        model.addConstrs(x.sum(i, "*", "*") == 1 for i in self.orders)
        # add constraints - flow conservation
        model.addConstrs(x.sum(i, "*", k) == x.sum("*", i, k) for i in self.vertices for k in self.vehicles)
        # add constraints - vehicle capacity
        for k in self.vehicles:
            model.addConstr(sum(self.order_dmds[i] * x[i, j, k] for i in self.orders for j in self.vertices) <=
                            self.veh_caps[k],
                            f"veh_{k}_cap")
        # add constraints - a vehicle can only start from (terminate at) its halting depot
        for i in self.depots:
            for k in self.vehicles:
                if k in self.depot_vehs[i]:
                    # vehicle k starts at depot i
                    model.addConstr(sum(x[i, j, k] for j in self.orders) <= 1, f"veh_{k}_s")
                else:
                    model.addConstr(sum(x[i, j, k] for j in self.orders) == 0, f"veh_{k}_ns")

        # add constraints - subtour elimination
        for sub_orders in powerset(self.orders, 1):
            for k in self.vehicles:
                model.addConstr(sum(x[i, j, k] for i in sub_orders for j in sub_orders) <= len(sub_orders) - 1,
                                "subtour")

        self.model = model
        self.flow_vars = x

    def run(self):
        super(GurobiSolver, self).run()
        self.model.optimize()

    def decode(self):
        depot_locs = self.vrp.asset.depots.get_attributes("location")
        order_unloading_locs = self.vrp.task.get_attributes("unloading_location")
        vertex2locs = {**depot_locs, **order_unloading_locs}
        route_plan = RoutePlan()
        for d in self.depots:
            for k in self.depot_vehs[d]:
                path_trace = {}
                for i, j in product(self.vertices, self.vertices):
                    if self.flow_vars[i, j, k].X == 1:
                        path_trace[i] = j
                pre_vertex = d
                route = Route([vertex2locs[d]])
                for _ in range(len(path_trace)):
                    next_vertex = path_trace[pre_vertex]
                    route.append(vertex2locs[next_vertex])
                    pre_vertex = next_vertex
                route_plan[k] = route
        self.best_solution = route_plan
        self._vrp.solution = self.best_solution
        return route_plan

    def feed(self):
        pass
