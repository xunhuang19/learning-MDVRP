import os
import time
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from expt.learning_cvrp.ga import GA
from expt.learning_cvrp.train import load_subproblem_data, SubproblemNetwork, restore
from expt.learning_cvrp.utils import *
import argparse
from vrpkit.vrp.formulation import load_instance_data
from vrpkit.vrp.hga import HGA
from vrpkit.vrp.insertion import NearestInsertionHeuristic


def run_lga_nni(vrps, process_vrps, mdvrp, model, route_solver, nni_solution):
    """
    Solve problem and calculate running time using the proposed LGA-NNI framework
    """
    assignment_solver = GA()
    assignment_solver.initialize(process_vrps, mdvrp, model)

    start_time = time.time()
    assignment_solver.run()
    collab_assignment = assignment_solver.best_solution.allocation
    tt_collab = get_solution(vrps, collab_assignment, route_solver)
    end_time = time.time()
    run_time = end_time - start_time

    if tt_collab > nni_solution:
        tt_collab = nni_solution
    return tt_collab, run_time


def run_benchmark(solver, vrp, initialize=True):
    start_time = time.time()
    if initialize:
        solver.solve(vrp)
    else:
        solver.run()
        solver.decode()
    end_time = time.time()
    run_time = end_time - start_time
    _, kpis = vrp.evaluate()
    tt_collab = kpis["Travel Time"]
    return tt_collab, run_time


def run_hga(vrp, args):
    solver = HGA()
    nni_solver = NearestInsertionHeuristic()
    nni_solver.config(depot_assignment_method="approx",
                      do_post_2opt=True,
                      post_2opt_iter_n=1,
                      do_post_opt_split=True)
    if args.feed:
        _, _ = run_benchmark(nni_solver, vrp)
        solver.initialize(vrp)
        solver.feed(nni_solver.best_solution)
        tt_collab, run_time = run_benchmark(solver, vrp, initialize=False)
    else:
        tt_collab, run_time = run_benchmark(solver, vrp)
    return tt_collab, run_time


def run_nni(vrp):
    solver = NearestInsertionHeuristic()
    solver.config(depot_assignment_method="approx",
                  do_post_2opt=True,
                  post_2opt_iter_n=1,
                  do_post_opt_split=True)
    tt_collab, run_time = run_benchmark(solver, vrp)
    return tt_collab, run_time


def generate_result(args, solve_instance, model):
    print(f'generating solution for instance {solve_instance}')
    args.instance_dir = args.dataset_dir / solve_instance

    # load instance data
    carrier_data, depot_data, vehicle_data, cust_data = load_data(args.instance_dir)
    vrps = load_instance_data(carrier_data, depot_data, vehicle_data, cust_data)
    vrp = sum(vrps)
    assets = [vrp.asset for vrp in vrps]  # each asset represent a logistics service provider
    total_comp_num = len(assets)
    print(f'loading data for instance {solve_instance}...')

    # get non-collaborative cost
    non_collab_assignment = []
    for i in range(total_comp_num):
        non_collab_assignment.append(list(cust_data[cust_data['carrier_id'] == i]['customer_id']))

    route_solver = NearestInsertionHeuristic()
    route_solver.config(depot_assignment_method="approx",
                        do_post_2opt=True,
                        post_2opt_iter_n=1,
                        do_post_opt_split=True)
    tt_no_collab = get_solution(vrps, non_collab_assignment, route_solver)

    if args.hga:
        hga_min_cost = []
        hga_time = []
        for i in range(args.n_result):  # run for n_result generations
            print(f'HGA Generation: {i}')
            hga_collab_cost, hga_run_time = zip(*Parallel(n_jobs=args.n_cpus)(
                delayed(run_hga)(vrp, args) for _ in range(args.n_gen)))  # get result for n_gen iteration
            hga_min_cost.append(min(hga_collab_cost))
            hga_time.append(np.mean(hga_run_time))
        hga_mean = np.mean(hga_min_cost)
        df_hga = pd.DataFrame(
            {'Instance': np.repeat(solve_instance, args.n_result), 'shuffle time': list(range(args.n_result)),
             'cost': hga_min_cost, 'run time': hga_time})
        df_hga.to_csv(args.result_save_dir / f'{solve_instance}_hga.csv', index=False)

    if args.nni:
        nni_min_cost = []
        nni_time = []
        for i in range(args.n_result):
            print(f'NNI Generation: {i}')
            nni_collab_cost, nni_run_time = zip(*Parallel(n_jobs=args.n_cpus)(
                delayed(run_nni)(vrp) for _ in range(args.n_gen)))
            nni_min_cost.append(min(nni_collab_cost))
            nni_time.append(np.mean(nni_run_time))
        nni_perf = 1 - (nni_min_cost - hga_mean) / hga_mean

        df_nni = pd.DataFrame(
            {'Instance': np.repeat(solve_instance, args.n_result), 'shuffle time': list(range(args.n_result)),
             'cost': nni_min_cost, 'performance': nni_perf, 'time': nni_time})
        df_nni.to_csv(args.result_save_dir / f'{solve_instance}_nni.csv', index=False)

    if args.lga_nni:
        # load processed data for model input
        depot_process, cust_process, _, _, _ = process_instance(depot_data, vehicle_data, cust_data)
        process_vrps = load_instance_data(carrier_data, depot_process, vehicle_data, cust_process)

        mdvrp = get_mdvrp(args.instance_dir)
        nni_solution, _ = run_nni(vrp)
        lga_min_cost = []
        lga_time = []
        for i in range(args.n_result):
            print(f'LGA Generation: {i}')
            lga_collab_cost, lga_run_time = zip(
                *Parallel(n_jobs=args.n_cpus)(
                    delayed(run_lga_nni)(vrps, process_vrps, mdvrp, model, route_solver, nni_solution) for _ in
                    range(args.n_gen)))
            lga_min_cost.append(min(lga_collab_cost))
            lga_time.append(np.mean(lga_run_time))

        lga_perf = 1 - (lga_min_cost - hga_mean) / hga_mean

        df_lga = pd.DataFrame(
            {'Instance': np.repeat(solve_instance, args.n_result), 'shuffle time': list(range(args.n_result)),
             'cost': lga_min_cost, 'performance': lga_perf, 'run time': lga_time})
        df_lga.to_csv(args.result_save_dir / f'{solve_instance}_lga.csv', index=False)

    return solve_instance, tt_no_collab


parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=Path)
parser.add_argument('instance_name', type=str)
parser.add_argument('train_dir', type=Path)
parser.add_argument('--lga_nni', action='store_true')
parser.add_argument('--hga', action='store_true')
parser.add_argument('--nni', action='store_true')
parser.add_argument('--feed', action='store_true')
parser.add_argument('--data_suffix', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_batch', type=int, default=2048)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--d_hidden', type=int, default=64)
parser.add_argument('--transformer_heads', type=int, default=4)
parser.add_argument('--loss', type=str, default='MAE')
parser.add_argument('--augment_rotate', action='store_true')
parser.add_argument('--augment_flip', action='store_true')
parser.add_argument('--fit_subproblem', action='store_true')
parser.add_argument('--n_steps', type=int, default=30000)
parser.add_argument('--save_dir', type=Path)
parser.add_argument('--eval_partition', type=str, default='val')
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--step', type=int, default=30000)
parser.add_argument('--n_gen', type=int, default=50)
parser.add_argument('--n_result', type=int, default=30)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--n_cpus', type=float, default=None)

type_map = {a.dest: a.type for a in parser._actions}

if __name__ == '__main__':
    start_time = time.time()
    args = parser.parse_args()
    args.result_save_dir = Path(f'data/results{args.n_gen}')
    args.result_save_dir.mkdir(parents=True, exist_ok=True)
    model_name = f'layer{args.n_layers}_head{args.transformer_heads}_lr{args.lr}_hidden{args.d_hidden}_batch{args.n_batch}_step{args.n_steps}'
    args.model_save_dir = args.train_dir / 'models' / model_name
    args.generate_save_dir = args.train_dir / 'generations'

    config = args.model_save_dir / 'config.yaml'
    assert config.exists()
    obj = load_yaml(config)
    for k, v in obj.items():
        if getattr(args, k) == parser.get_default(k):
            type_ = type_map[k]
            setattr(args, k, type_(v) if type_ is not None else v)
    print(f'Loaded args from {config}')

    # load learned model
    path_eval = args.save_dir / f'{args.eval_partition}{args.data_suffix}_subproblems.npz'
    d_eval = load_subproblem_data(path_eval)
    model = SubproblemNetwork(args, d_eval)
    restore(args, model)
    model = model.to(args.device)

    instance, no_collab_cost = zip(
        *Parallel(n_jobs=args.n_cpus)(delayed(generate_result)(args, i, model) for i in os.listdir(args.dataset_dir)))

    df_benefit = pd.DataFrame({'Instance': instance, 'Non-collaborative cost': no_collab_cost})
    df_benefit.to_csv(args.result_save_dir / f'benefit_{args.instance_name}csv', index=False)
