from collabvrp.formulation import load_instance
from collabvrp.mechanism import IterativeAuctionBasedMechanism as IAM
from expt.scalable_cvrp.utils import proportionally_sampled_assignment

from vrpkit.vrp.insertion import NearestInsertionHeuristic as NIH


class InstanceGenerator:

    def __init__(self, base_vrp):
        self.base_instance = base_vrp


def average_chunking():
    pass


def instance_convertor():
    """generate multi-carrier routing instance given """

    # generate subinstances
    base_instance_folder = "./data/instances/benchmark/Solomon/R101"
    base_covrp = load_instance(base_instance_folder)
    n_carriers = 4
    method = "random"
    method = "sequential"
    n_vehicles = len(base_covrp.asset.fleet)
    if n_carriers > n_vehicles:
        raise ValueError("too many carriers, each carrier must have at least one vehicle")

    all_vehicle_ids = base_covrp.asset.fleet.keys_in_list()
    if method == "sequential":
        pass


def main():
    instance_folder = "./data/instances/small/R_20_4"
    routing_solver = NIH()

    results = {"iter num": {}, "total gain": {}, "system profit": {}, "time usage": {}}

    covrp = load_instance(instance_folder)
    covrp.set_carrier_solver(routing_solver)

    deviation_strategies = ["BBSTD", "DSIBSTD", "DSIBETD", "OCIBSTD", "OCIBETD"]
    mechanisms = [IAM(deviation_method=s) for s in deviation_strategies]
    for kpi in results:
        for s in deviation_strategies:
            results[kpi][s] = []

    for s, mechanism in zip(deviation_strategies, mechanisms):
        mechanism.solve(covrp)
        results["iter num"][s].append(len(mechanism.logbook["iteration"]))
        results["total gain"][s].append(sum(mechanism.logbook["gain"]))
        results["system profit"][s].append(mechanism.logbook["system profit"][-1])
        results["time usage"][s].append(sum(mechanism.logbook["time usage"]))

    print(results)


def test():
    l = list(range(10))
    k = 4
    n = 10
    for assign in proportionally_sampled_assignment(l, k, n):
        print(assign)


if __name__ == "__main__":
    test()
