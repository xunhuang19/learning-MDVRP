import argparse
import os
from pathlib import Path
import numpy as np
from joblib import delayed, Parallel
from expt.learning_cvrp.utils import MDVRProblem, MDVRSubProblem


# process inputs for training
def preprocess(args, i):
    s = np.load(args.partition_dir / f'{i}', allow_pickle=True)
    allocations, dists = s['assignments'], s['dists']
    p = MDVRProblem(s['xys'], s['demands'], s['capacities'], s['n_depot'])
    node_features = p.get_node_features()
    all_node_idxs = []

    for idx in range(len(dists)):
        subp = MDVRSubProblem(p, allocations[idx], dists[idx])
        all_node_idxs.append(subp.get_node_idx())
    n_subp_nodes = [len(node_idxs) for node_idxs in all_node_idxs]
    return node_features, np.concatenate(all_node_idxs), np.array(n_subp_nodes), dists, allocations


parser = argparse.ArgumentParser()
parser.add_argument('save_dir', type=Path)
parser.add_argument('--n_cpus', type=int, default=12)
parser.add_argument('--partition', type=str, choices=['train', 'val', 'test'])

if __name__ == "__main__":
    args = parser.parse_args()
    args.partition_dir = partition_dir = args.save_dir / args.partition
    output_name = args.partition + '_subproblems'

    n_instances = len([x for x in partition_dir.glob('*.npz')])
    print(f'Preprocessing {n_instances} instances from partition {args.partition} into {output_name}.npz', flush=True)

    node_features, all_node_idxs, n_subp_nodes, dists, allocations = zip(
        *Parallel(n_jobs=args.n_cpus)(delayed(preprocess)(args, i) for i in os.listdir(args.partition_dir)))

    out_path = args.save_dir / f'{output_name}.npz'

    n_nodes = np.array([nf.shape[0] for nf in node_features])
    prob_offsets = np.cumsum(n_nodes) - n_nodes

    node_offsets = [[o] * len(ns) for o, ns in zip(prob_offsets, n_subp_nodes)]

    node_features, node_offsets, all_node_idxs, n_subp_nodes, dists, allocations = map(np.concatenate,
                                                                                       [node_features, node_offsets,
                                                                                        all_node_idxs, n_subp_nodes,
                                                                                        dists, allocations
                                                                                        ])

    np.savez(out_path, xs=node_features, offsets=node_offsets, subp_node_idxs=all_node_idxs, n_subp_nodes=n_subp_nodes,
             dists=dists, allocations=allocations)
