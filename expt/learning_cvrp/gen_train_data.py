from expt.utils import random_assignment
from vrpkit.vrp.formulation import load_instance, MDVRPTW
from vrpkit.vrp.hga import HGA
import pandas as pd


def data_gen(
        instance_folder="../../data/instances/large/P_200_10",
        save_data_folder=f"../../data/results/train_data_test",
        sample_size=5,
):
    vrps = load_instance(instance_folder)
    central_vrp = sum(vrps)
    assets = [vrp.asset for vrp in vrps]  # each asset represent a logistics service provider
    all_order_ids = central_vrp.task.keys_in_list()
    total_comp_num = len(assets)
    total_cust_num = len(all_order_ids)
    total_asgn_num = total_comp_num ** total_cust_num
    if sample_size >= total_asgn_num:
        sample_size = total_asgn_num

    solver = HGA()
    results = {"assignment": [], "travel time": []}
    for _ in range(sample_size):
        assignment = random_assignment(all_order_ids, total_comp_num)
        while assignment in results["assignment"]:
            assignment = random_assignment(all_order_ids, total_comp_num)
        tt = 0
        for i in range(total_comp_num):
            task = central_vrp.task.subset(assignment[i])
            if not task.is_empty():
                vrp = MDVRPTW(assets[i], task, central_vrp.cost_matrix, central_vrp.objective)
                solver.solve(vrp)
                _, metrics = vrp.evaluate()
                tt += metrics["Travel Time"]

        results["assignment"].append(assignment)
        results["travel time"].append(tt)
    pd.DataFrame(results).to_csv(save_data_folder, index=False)


if __name__ == "__main__":
    data_gen()
