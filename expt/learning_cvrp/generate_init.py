import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from expt.learning_cvrp.utils import get_cost
from expt.utils import random_feasible_assignment
from vrpkit.vrp.formulation import *
from vrpkit.vrp.insertion import NearestInsertionHeuristic


def generate_data(args, instance_name):
    # load and process instance
    instance_dir = args.dataset_dir / instance_name
    carrier_data, depot_data, vehicle_data, cust_data = load_data(instance_dir)
    n_depot = len(depot_data)
    depot_process, cust_process, xys, demands, capacities = process_instance(depot_data, vehicle_data, cust_data)

    vrps = load_instance_data(carrier_data, depot_process, vehicle_data, cust_process)
    solver = NearestInsertionHeuristic()
    solver.config(depot_assignment_method="approx",
                  do_post_2opt=True,
                  post_2opt_iter_n=1,
                  do_post_opt_split=True)

    assignments = random_feasible_assignment(vrps, args.n_sample)
    if args.n_sample > len(assignments):
        args.n_sample = len(assignments)
        print(f'sampled assignments for instance {instance_name} are more than the available feasible assignment')
    dists = []

    for n in range(min(len(assignments), args.n_sample)):
        assignment = assignments[n]
        tt = get_cost(vrps, assignment, solver)
        print(f'Generating solution for assignment {n} in instance {instance_name}')
        dists.append(tt)

    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    # train is now 80% of the entire data set
    assignments_train, assignments_test, dists_train, dists_test = train_test_split(assignments, dists,
                                                                                    test_size=1 - train_ratio)

    print(f'Generating {len(dists_train)} allocations from instance {instance_name} to partition train')

    # test is now 10% of the initial data set
    # validation is now 10% of the initial data set
    assignments_val, assignments_test, dists_val, dists_test = train_test_split(assignments_test, dists_test,
                                                                                test_size=test_ratio / (
                                                                                        test_ratio + validation_ratio))

    print(f'Generating {len(dists_test)} allocations from instance {instance_name} to partition test')
    print(f'Generating {len(dists_val)} allocations from instance {instance_name} to partition val')

    args.train_dir.mkdir(exist_ok=True)
    np.savez(args.train_dir / f'{instance_name}_train.npz',
             xys=xys,
             demands=demands,
             capacities=capacities,
             dists=dists_train,
             assignments=np.array(assignments_train, dtype=object),
             n_depot=n_depot
             )

    args.test_dir.mkdir(exist_ok=True)
    np.savez(args.test_dir / f'{instance_name}_test.npz',
             xys=xys,
             demands=demands,
             capacities=capacities,
             dists=dists_test,
             assignments=np.array(assignments_test, dtype=object),
             n_depot=n_depot
             )

    args.val_dir.mkdir(exist_ok=True)
    np.savez(args.val_dir / f'{instance_name}_val.npz',
             xys=xys,
             demands=demands,
             capacities=capacities,
             dists=dists_val,
             assignments=np.array(assignments_val, dtype=object),
             n_depot=n_depot
             )


parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=Path)
parser.add_argument('save_dir', type=Path)
parser.add_argument('--instance_name', type=str)
parser.add_argument('--n_sample', type=int, default=5000)
parser.add_argument('--n_cpus', type=int, default=12)
parser.add_argument('--n_customer', type=int, default=100)

if __name__ == "__main__":
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.train_dir = args.save_dir / 'train'
    args.test_dir = args.save_dir / 'test'
    args.val_dir = args.save_dir / 'val'
    Parallel(n_jobs=args.n_cpus)(delayed(generate_data)(args, i) for i in os.listdir(args.dataset_dir))
