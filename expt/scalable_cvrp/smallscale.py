# small-scale experiments:
# (1) generate 20 instances with 8 customers and 2 companies from R101
# (2) use exact solver to calculate
#       non-collaborated operation,
#       collaborated operation,
#       the worst equilibrium,
#       centrally optimised equilibrium.
# (3) change to heuristics to calculate "collaborated operation"


from vrpkit.base.costmatrix import CostMatrix
from vrpkit.base.formulation import VRP
from vrpkit.base.objective import Objective, TravelTime
from vrpkit.vrp.formulation import load_instance
from vrpkit.vrp.hga import HGA
from expt.utils import all_assignments, is_equilibrium

import random

import pandas as pd
import matplotlib.pyplot as plt


def add_prefix():
    import pandas as pd
    inst_folder = "./data/instances/middle/R_100_4"
    depot_data = pd.read_csv(inst_folder + "/depots.csv")
    vehicle_data = pd.read_csv(inst_folder + "/vehicles.csv")
    cust_data = pd.read_csv(inst_folder + "/customers.csv")

    depot_data["depot_id"] = depot_data["depot_id"].apply(lambda i: "d" + str(i))
    vehicle_data["depot_id"] = vehicle_data["depot_id"].apply(lambda i: "d" + str(i))
    vehicle_data["vehicle_id"] = vehicle_data["vehicle_id"].apply(lambda i: "v" + str(i))
    cust_data["customer_id"] = cust_data["customer_id"].apply(lambda i: "c" + str(i))

    depot_data.to_csv(inst_folder + "/depots.csv", index=None)
    vehicle_data.to_csv(inst_folder + "/vehicles.csv", index=None)
    cust_data.to_csv(inst_folder + "/customers.csv", index=None)


def main():
    random.seed(100)
    base_inst_folder = "../data/instances/middle/R_100_4"
    base_vrp = sum(load_instance(base_inst_folder))
    base_order_ids = base_vrp.task.keys_in_list()
    solver = HGA()  # GurobiSolver()
    solver.config(pop_size=20, gen_n=10)
    objective = Objective([TravelTime()], [-1])

    asset1 = base_vrp.asset.subset(depot_ids="d0", vehicle_ids="v0")
    asset2 = base_vrp.asset.subset(depot_ids="d0", vehicle_ids="v1")

    inst_num = 20
    for i in range(inst_num):
        sampled_order_ids = random.sample(base_order_ids, 8)
        sampled_orders = base_vrp.task.subset(sampled_order_ids)
        cost_matrix = CostMatrix()
        cost_matrix.init_given_asset_and_task(asset1 + asset2, sampled_orders)

        res = {"comapny1": [], "company2": [], "total": [], "is_equilibrium": []}
        for assign in all_assignments(sampled_order_ids, 2):
            vrp1 = VRP(asset1, base_vrp.task.subset(assign[0]), cost_matrix, objective)
            vrp2 = VRP(asset1, base_vrp.task.subset(assign[1]), cost_matrix, objective)
            for vrp in [vrp1, vrp2]:
                solver.initialize(vrp)
                solver.run()
                solver.decode()
            obj_1 = vrp1.evaluate()[0]
            obj_2 = vrp2.evaluate()[0]
            res["comapny1"].append(obj_1)
            res["company2"].append(obj_2)
            res["total"].append(obj_1 + obj_2)
            res["is_equilibrium"].append(is_equilibrium(vrp1, vrp2, solver))

        pd.DataFrame(res).to_csv(f"../data/results/small_expt_br_{i}.csv")


def visual_result():
    summary = {"worst non-collab": [],
               "worst equ": [],
               "best equ": []}
    inst_nums = list(range(40))
    for i in inst_nums:
        data = pd.read_csv(f"../data/results/small_expt_{i}.csv")
        summary["worst non-collab"].append(min(data["total"]))
        summary["worst equ"].append(min(data[data["is_equilibrium"]]["total"]))
        summary["best equ"].append(max(data[data["is_equilibrium"]]["total"]))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(inst_nums, summary["worst non-collab"], label="Worst Non-collaboration", color="orange")
    ax.plot(inst_nums, summary["best equ"], label="Upper Bound", linestyle='dashed', color="blue")
    ax.plot(inst_nums, summary["worst equ"], label="Lower Bound", linestyle='dashed', color="blue")
    ax.fill_between(inst_nums, summary["worst equ"], summary["best equ"], alpha=0.3, color="green")

    ax.set_title("Mechanism Performance Under Perfect Decisions")
    ax.set_ylabel("-Travel Cost")
    ax.set_xlabel("Problem Instances")
    ax.legend()
    fig.savefig("../data/results/small_expt_res.png", dpi=480)


def compare_br_and_exact():
    exact_summary = {"worst non-collab": [],
                     "worst equ": [],
                     "best equ": []}
    br_summary = {"worst non-collab": [],
                  "worst equ": [],
                  "best equ": []}
    inst_nums = list(range(20))
    for i in inst_nums:
        data = pd.read_csv(f"../data/results/small_expt_{i}.csv")
        exact_summary["worst non-collab"].append(min(data["total"]))
        exact_summary["worst equ"].append(min(data[data["is_equilibrium"]]["total"]))
        exact_summary["best equ"].append(max(data[data["is_equilibrium"]]["total"]))
        data = pd.read_csv(f"../data/results/small_expt_br_{i}.csv")
        br_summary["worst non-collab"].append(min(data["total"]))
        br_summary["worst equ"].append(min(data[data["is_equilibrium"]]["total"]))
        br_summary["best equ"].append(max(data[data["is_equilibrium"]]["total"]))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(inst_nums, exact_summary["best equ"], label="Upper Bound-exact", linestyle='dashed', color="blue")
    ax.plot(inst_nums, exact_summary["worst equ"], label="Lower Bound-exact", linestyle='dashed', color="blue")
    ax.fill_between(inst_nums, exact_summary["worst equ"], exact_summary["best equ"], alpha=0.3, color="blue")

    ax.plot(inst_nums, br_summary["best equ"], label="Upper Bound-br", linestyle='dashed', color="orange")
    ax.plot(inst_nums, br_summary["worst equ"], label="Lower Bound-br", linestyle='dashed', color="orange")
    ax.fill_between(inst_nums, br_summary["worst equ"], br_summary["best equ"], alpha=0.3, color="orange")

    ax.set_title("Mechanism Comparison")
    ax.set_ylabel("-Travel Cost")
    ax.set_xlabel("Problem Instances")
    ax.legend()
    fig.savefig("../data/results/small_expt_compare.png", dpi=480)


if __name__ == "__main__":
    compare_br_and_exact()
