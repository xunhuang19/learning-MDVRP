import numpy as np
from sklearn import preprocessing

from ..base.formulation import VRP
from ..base.facility import Asset, DepotServicePoint, Depot, Vehicle
from ..base.demand import Task, UnloadingPoint, Order
from ..base.costmatrix import CostMatrix
from ..base.objective import Objective, GrossRevenue, TravelTime, ServiceDelay, IdleTime, OverLoad

import pandas as pd


class CVRP(VRP):
    """Single-depot Capacitated Vehicle Routing Problems"""

    def validate_data(self):
        super().validate_data()
        pass


class MDVRPTW(VRP):
    """Multi-depot Vehicle Routing Problem with Time Windows"""

    def validate_data(self):
        super().validate_data()
        pass


def load_data(instance_data_folder="./data/instances/t_10_2"):
    carrier_data = pd.read_csv(instance_data_folder / "carriers.csv")
    depot_data = pd.read_csv(instance_data_folder / "depots.csv")
    vehicle_data = pd.read_csv(instance_data_folder / "vehicles.csv")
    cust_data = pd.read_csv(instance_data_folder / "customers.csv")
    return carrier_data, depot_data, vehicle_data, cust_data


def process_instance(depot_data, vehicle_data, cust_data):
    capacities = vehicle_data['capacity']
    demands = pd.concat([pd.DataFrame(np.zeros((len(depot_data), 1))), cust_data['demand']])
    xys = pd.concat([depot_data[["long", "lat"]], cust_data[["long", "lat"]]])
    scaler = xys.max().max()
    xys_scaled = (xys/scaler).to_numpy()

    depot_data['long'] = xys_scaled[0:len(depot_data), 0]
    depot_data['lat'] = xys_scaled[0:len(depot_data), 1]
    cust_data['long'] = xys_scaled[len(depot_data):, 0]
    cust_data['lat'] = xys_scaled[len(depot_data):, 1]

    return depot_data, cust_data, xys_scaled, demands, capacities


def load_instance_data(carrier_data, depot_data, vehicle_data, cust_data):
    """load multi-depot VRP instances from data files.
    obligatory data files: carriers.csv; depots.csv; vehicles.csv; customers.csv
    """

    # carrier_data = pd.read_csv(instance_data_folder + "/carriers.csv")
    # depot_data = pd.read_csv(instance_data_folder + "/depots.csv")
    # vehicle_data = pd.read_csv(instance_data_folder + "/vehicles.csv")
    # cust_data = pd.read_csv(instance_data_folder + "/customers.csv")

    assets = []
    tasks = []
    for carrier_id in carrier_data["carrier_id"]:
        # load depots and vehicles info
        asset = Asset()
        carrier_depot_data = depot_data[depot_data["carrier_id"] == carrier_id]
        for _, d_row in carrier_depot_data.iterrows():
            depot_id = d_row["depot_id"]
            depot_loc = DepotServicePoint(depot_id, long=d_row["long"], lat=d_row["lat"])
            asset.depots[depot_id] = Depot(depot_id, location=depot_loc)
            depot_vehicle_data = vehicle_data[vehicle_data["depot_id"] == depot_id]
            for _, v_row in depot_vehicle_data.iterrows():
                vehicle_id = v_row["vehicle_id"]
                asset.fleet[vehicle_id] = Vehicle(vehicle_id, v_row["capacity"], v_row["route_limit"])
                asset.halt[vehicle_id] = depot_id

        # load customer - delivery orders - info
        task = Task()
        carrier_cust_data = cust_data[cust_data["carrier_id"] == carrier_id]
        for _, row in carrier_cust_data.iterrows():
            customer_id = row["customer_id"]
            ulp = UnloadingPoint(customer_id, long=row["long"], lat=row["lat"], tw_start=row["tw_start"],
                                 tw_end=row["tw_end"])
            task[customer_id] = Order(customer_id, row["demand"], row["value"], unloading_location=ulp)

        assets.append(asset)
        tasks.append(task)

    # create a global cost matrix
    cost_matrix = CostMatrix()
    cost_matrix.init_given_asset_and_task(sum(assets), sum(tasks))

    # assemble routing instances
    instances = []
    metrics = [GrossRevenue(), TravelTime(), ServiceDelay(), IdleTime(), OverLoad()]
    for asset, task in zip(assets, tasks):
        objective = Objective(metrics, [1, -1, 0, 0, -1000])
        vrp = MDVRPTW(asset, task, cost_matrix, objective)
        instances.append(vrp)

    return instances


def load_instance(instance_data_folder):
    """load multi-depot VRP instances from data files.
    obligatory data files: carriers.csv; depots.csv; vehicles.csv; customers.csv
    """

    carrier_data = pd.read_csv(instance_data_folder + "/carriers.csv")
    depot_data = pd.read_csv(instance_data_folder + "/depots.csv")
    vehicle_data = pd.read_csv(instance_data_folder + "/vehicles.csv")
    cust_data = pd.read_csv(instance_data_folder + "/customers.csv")

    assets = []
    tasks = []
    for carrier_id in carrier_data["carrier_id"]:
        # load depots and vehicles info
        asset = Asset()
        carrier_depot_data = depot_data[depot_data["carrier_id"] == carrier_id]
        for _, d_row in carrier_depot_data.iterrows():
            depot_id = d_row["depot_id"]
            depot_loc = DepotServicePoint(depot_id, long=d_row["long"], lat=d_row["lat"])
            asset.depots[depot_id] = Depot(depot_id, location=depot_loc)
            depot_vehicle_data = vehicle_data[vehicle_data["depot_id"] == depot_id]
            for _, v_row in depot_vehicle_data.iterrows():
                vehicle_id = v_row["vehicle_id"]
                asset.fleet[vehicle_id] = Vehicle(vehicle_id, v_row["capacity"], v_row["route_limit"])
                asset.halt[vehicle_id] = depot_id

        # load customer - delivery orders - info
        task = Task()
        carrier_cust_data = cust_data[cust_data["carrier_id"] == carrier_id]
        for _, row in carrier_cust_data.iterrows():
            customer_id = row["customer_id"]
            ulp = UnloadingPoint(customer_id, long=row["long"], lat=row["lat"], tw_start=row["tw_start"],
                                 tw_end=row["tw_end"])
            task[customer_id] = Order(customer_id, row["demand"], row["value"], unloading_location=ulp)

        assets.append(asset)
        tasks.append(task)

    # create a global cost matrix
    cost_matrix = CostMatrix()
    cost_matrix.init_given_asset_and_task(sum(assets), sum(tasks))

    # assemble routing instances
    instances = []
    metrics = [GrossRevenue(), TravelTime(), ServiceDelay(), IdleTime(), OverLoad()]
    for asset, task in zip(assets, tasks):
        objective = Objective(metrics, [1, -1, 0, 0, -100])
        vrp = MDVRPTW(asset, task, cost_matrix, objective)
        instances.append(vrp)

    return instances
