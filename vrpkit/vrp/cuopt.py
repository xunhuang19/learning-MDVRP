from vrpkit.base.solver import Solver
from vrpkit.base.solution import RoutePlan, Route

import requests

# todo: check server
class CuOptSolver(Solver):
    """a wrapper of NVIDIA CuOpt

    Sever IP: "155.198.89.219", Container Port: "5000"

    """

    def __init__(self, vrp=None, ip="127.0.0.1", port="5000", time_limit=10, n_climbers=128):
        super(CuOptSolver, self).__init__(vrp)
        self.depots = []
        self.orders = []
        self.order_dmds = []
        self.vehicles = []
        self.veh_caps = []
        self.veh_halts = []
        self.distance_matrix = []
        self._ip = ip
        self._port = port
        self._url = f"http://{self._ip}:{self._port}/cuopt/"
        self.time_limit = time_limit
        self.n_climbers = n_climbers
        self.server_resp = None  # {'status':.,'num_vehicles':.,'vehicle_data':{'veh_id':{'route':[]}}}
        self.encode()

    @property
    def ip(self):
        return self._ip

    @ip.setter
    def ip(self, ip):
        self._ip = ip
        self._url = f"http://{self._ip}:{self._port}/cuopt/"

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, port):
        self._port = port
        self._url = f"http://{self._ip}:{self._port}/cuopt/"

    def check_server_status(self):
        if requests.get(self._url + "health").status_code != 200:
            raise ValueError(f"There is no CuOpt Server running on {self._url}")

    def encode(self):
        # CuOpt requires all depots and orders ids to be integers
        orig_depot_ids = self.vrp.asset.depots.keys_in_list()
        orig_order_ids = self.vrp.task.keys_in_list()
        n_depots = len(orig_depot_ids)
        n_orders = len(orig_order_ids)
        self.depots = list(range(n_depots))
        self.orders = list(range(n_depots, n_orders + n_depots))
        self.order_dmds = list(self.vrp.task.get_attributes("volume").values())
        self.vehicles = self.vrp.asset.fleet.keys_in_list()
        self.veh_caps = list(self.vrp.asset.fleet.get_attributes("capacity").values())
        self.veh_halts = []
        for veh in self.vehicles:
            halt_depo = self.vrp.asset.halt[veh]
            halt_depo_int = orig_depot_ids.index(halt_depo)
            self.veh_halts.append([halt_depo_int, halt_depo_int])
        orig_all_node_ids = []
        for d in orig_depot_ids:
            orig_all_node_ids.append(self.vrp.asset.depots[d].location.node_id)
        for i in orig_order_ids:
            orig_all_node_ids.append(self.vrp.task[i].unloading_location.node_id)
        self.distance_matrix = [[self.vrp.cost_matrix[i][j] for j in orig_all_node_ids] for i in orig_all_node_ids]

    def initialize(self, vrp=None, *args, **kwargs):
        super(CuOptSolver, self).initialize(vrp)
        # Set the cost matrix
        data_params = {"return_data_state": False}
        cost_data = {"cost_matrix": {0: self.distance_matrix}}
        response_set = requests.post(self._url + "set_cost_matrix", params=data_params, json=cost_data)
        assert response_set.status_code == 200, print(response_set.json())

        # set task data
        task_data = {"task_locations": self.orders, "demand": [self.order_dmds]}
        response_set = requests.post(self._url + "set_task_data", json=task_data)
        assert response_set.status_code == 200, print(response_set.json())

        # set fleet data
        fleet_data = {"vehicle_locations": self.veh_halts,
                      "vehicle_ids": self.vehicles,  # require veh id to be str
                      "capacities": [self.veh_caps]}
        response_set = requests.post(self._url + "set_fleet_data", json=fleet_data)
        assert response_set.status_code == 200, print(response_set.json())

        # set solver configuration
        solver_config = {"time_limit": self.time_limit, "number_of_climbers": self.n_climbers}
        response_set = requests.post(self._url + "set_solver_config", json=solver_config)
        assert response_set.status_code == 200, print(response_set.json())

    def run(self, *args, **kwargs):
        super(CuOptSolver, self).run()
        # Solve the problem
        solver_response = requests.get(self._url + "get_optimized_routes")
        assert solver_response.status_code == 200, print(solver_response.json())
        # Process returned data
        solver_resp = solver_response.json()["response"]["solver_response"]
        self.server_resp = solver_resp

    def decode(self):
        if self.server_resp["status"] == 0:
            veh_data = self.server_resp["vehicle_data"]
            route_plan = RoutePlan()
            n_depots = len(self.depots)
            orig_depots_ids = self.vrp.asset.depots.keys_in_list()
            orig_order_ids = self.vrp.task.keys_in_list()
            for veh_id in veh_data:
                route_data = veh_data[veh_id]["route"]
                route = Route()
                for vertex_i in route_data:
                    if vertex_i < n_depots:
                        orig_id = orig_depots_ids[vertex_i]
                        route.append(self.vrp.asset.depots[orig_id].location)
                    else:
                        orig_id = orig_order_ids[vertex_i - n_depots]
                        route.append(self.vrp.task[orig_id].unloading_location)
                if len(route) > 0:
                    route_plan[veh_id] = route
            self.best_solution = route_plan
        else:
            print("NVIDIA cuOpt Failed to find a solution with status : ", self.server_resp["status"])

        self._vrp.solution = self.best_solution
        return self.best_solution
