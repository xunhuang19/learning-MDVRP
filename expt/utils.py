import math
import random
from itertools import product, combinations
from typing import Sequence, Iterable

import numpy as np
from numpy import dot
from numpy.linalg import norm

from vrpkit.base.constant import INF
from vrpkit.base.demand import Task
from vrpkit.base.formulation import VRP
from vrpkit.base.objective import SpareLoad
from vrpkit.base.solver import Solver


def all_assignments(l: Sequence, k: int):
    """Enumerate all possible assignments that allocate each element in l to one of k clusters labels

    (1) the number of all possible assignments are (k)**l
    (2) sequence matters. i.e., for l=[1,2,3] and k=2, "[1], [2,3]" and "[2, 3], [1]" are different.

    """
    for labels in product(range(k), repeat=len(l)):
        partition = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            partition[label].append(l[i])
        yield partition


def random_assignments(l, k, n):
    """randomly sample n assignments from
    all possible assignments that allocate each element in l to one of k clusters labels
    """
    for _ in range(n):
        partition = random_assignment(l, k)
        yield partition


def random_assignment(l, k):
    """randomly sample one assignment from
    all possible assignments that allocate each element in l to one of k clusters labels
    """

    partition = [[] for _ in range(k)]
    labels = random.choices(range(k), k=len(l))
    for i, label in enumerate(labels):
        partition[label].append(l[i])
    return partition


def random_feasible_assignment(vrps, n):
    central_vrp = sum(vrps)
    assets = [vrp.asset for vrp in vrps]  # each asset represent a logistics service provider
    all_order_ids = central_vrp.task.keys_in_list()
    total_comp_num = len(assets)
    total_cust_num = len(all_order_ids)
    total_asgn_num = total_comp_num ** total_cust_num

    valid_n = 0
    total_n = 0
    assignments = []
    while valid_n < n and total_n < total_asgn_num:
        assignment = random_assignment(all_order_ids, total_comp_num)
        total_n += 1
        while assignment in assignments:
            assignment = random_assignment(all_order_ids, total_comp_num)
        for i in range(total_comp_num):
            depot_vehicles = assets[i].depot_vehicles()
            max_cap = sum([v.capacity for _, vehicles in depot_vehicles.items() for v in vehicles])
            order_vol = sum([central_vrp.task.data[order_id].volume for order_id in assignment[i]])
            if order_vol > max_cap:
                break
            elif i == total_comp_num-1:
                assignments.append(assignment)
                valid_n += 1
    return assignments


def random_distribution_assignment(l, k, d):
    """sample one assignment that allocate each element in l to one of k clusters labels,
    with the probability of each k being selected following distribution d
    """

    partition = [[] for _ in range(k)]
    labels = random.choices(range(k), weights=d, k=len(l))
    for i, label in enumerate(labels):
        partition[label].append(l[i])
    return partition


def evaluate_marginal_contribution(orders: Task, vrp: VRP, solver: Solver, accel=False):
    """Evaluate the marginal contribution of a bundle of orders (task) to a solution's performance (route plan).
    If orders have been included in the route plan, the marginal contribution is VRP(current) - VRP(current-orders).
    If orders are new to existing orders, the marginal contribution is VRP(current+orders) - VRP(current).
    If the spare capacity is not enough for inclusion, the returned value is negative infinity
    Any other situations is not applicable and will yield errors.

    Args
    accel: if true, the evaluation will be accelerated by a simple removal when possible

    """

    intersect_orders = orders.intersection(vrp.task)

    if 0 < len(intersect_orders) < len(orders):
        raise ValueError("Input Error: The orders to be evaluated are partially intersected with the existing orders.")

    orig_obj, _ = vrp.evaluate()

    if intersect_orders.is_empty():
        # case 1: the orders to be evaluated are completely new.
        if orders.total_volume() > SpareLoad.evaluate(vrp.solution, vrp.cost_matrix, vrp.asset, vrp.task):
            return -INF
        new_vrp = VRP(vrp.asset, vrp.task + orders, vrp.cost_matrix, vrp.objective)
        solver.solve(new_vrp)
        new_obj, _ = new_vrp.evaluate()
        obj_marg = new_obj - orig_obj
    else:
        # case 2: the orders to be evaluated have already been included.
        if accel:
            reduced_solution = vrp.solution.copy()
            for order in orders.values():
                reduced_solution.remove(order.unloading_location)
            reduced_obj, _ = vrp.evaluate(reduced_solution)
        else:
            reduced_vrp = VRP(vrp.asset, vrp.task - orders, vrp.cost_matrix, vrp.objective)
            solver.solve(reduced_vrp)
            reduced_obj, _ = reduced_vrp.evaluate()
        obj_marg = orig_obj - reduced_obj

    return obj_marg


def is_equilibrium(vrps: Iterable[VRP], solver: Solver):
    """check whether an allocation is an equilibrium"""
    is_equ = True
    for current_vrp, other_vrp in product(vrps, vrps):
        if current_vrp is other_vrp:
            continue
        for order_id in current_vrp.task:
            sub_task = current_vrp.task.subset(order_id)
            current_marg = evaluate_marginal_contribution(sub_task, current_vrp, solver, True)
            opposite_marg = evaluate_marginal_contribution(sub_task, other_vrp, solver)
            if opposite_marg > current_marg:
                is_equ = False
    return is_equ


def integer_portion_split(portions, n):
    """divide n into len(portions) parts given the portions where sum(portions) equals 1"""
    splits = [p * n for p in portions]
    splits_int = [0 for _ in portions]

    sorted_indices = [i for i, x in sorted(enumerate(splits), key=lambda x: x[1], reverse=True)]
    for i in sorted_indices:
        splits_int[i] = min(max(1, int(splits[i] + 0.5)), n - sum(splits_int))

    if sum(splits_int) < n:
        gaps = [s_int - s for s_int, s in zip(splits_int, splits)]
        sorted_indices = [i for i, x in sorted(enumerate(gaps), key=lambda x: x[1])]
        for i in sorted_indices:
            splits_int[i] += 1
            if sum(splits_int) >= n:
                break
    return splits_int


def interger_equal_split(q, n):
    """equally divide n into q partitions"""
    weight = int(n / q + 0.5)
    if weight < 1:
        for i in range(q):
            if i < n:
                yield 1
            else:
                yield 0
    else:
        assigned = 0
        for i in range(q):
            if assigned + weight <= n:
                assigned += weight
                yield weight
            else:
                reduce_weight = n - assigned
                assigned += reduce_weight
                yield reduce_weight


def proportionally_sampled_assignment(l: Sequence, k: int, n):
    """proportionally sample n assignments from all possible assignments that allocate each element
    in l to one of k clusters labels
    """

    all_assign_num = k ** len(l)
    sub_categories = list(range(1, k + 1))
    C = {i: math.comb(k, i) for i in range(1, k + 1)}
    S = {}
    for i in range(1, k + 1):
        if i == 1:
            S[i] = 1
        else:
            S[i] = (i ** len(l) - 1) - sum(S[j] * math.comb(i, j) for j in range(1, i))

    sub_cate_portions = [C[i] * S[i] / all_assign_num for i in sub_categories]
    sub_cate_weights = integer_portion_split(sub_cate_portions, n)

    n_elements = len(l)
    splittable_indices = range(1, n_elements)
    for i in sub_categories:
        final_cate_weights = interger_equal_split(C[i], sub_cate_weights[i - 1])
        for combin, weight in zip(combinations(range(k), i), final_cate_weights):
            if weight > 0:
                print(f"combinations:{combin}, weights {weight}")
            sampled_partitions = []
            n_sampled = 0
            n_delimiters = len(combin) - 1
            while n_sampled < weight:
                partition = [[] for _ in range(k)]
                rand_perm = random.sample(l, n_elements)
                delimiter_locs = random.sample(splittable_indices, n_delimiters)
                delimiter_locs = [0] + sorted(delimiter_locs)
                delimiter_locs.append(None)

                if n_delimiters == 0:
                    partition[combin[0]] = list(rand_perm)
                else:
                    for j, s, e in zip(combin, delimiter_locs[:-1], delimiter_locs[1:]):
                        partition[j] = rand_perm[s:e]

                if partition in sampled_partitions:
                    continue
                else:
                    sampled_partitions.append(partition)
                    n_sampled += 1
                    yield partition


def generate_random_distribution(k):
    """
    create a random probability distribution of size k that sums up to 1
    """
    d = np.random.random(k)
    d /= d.sum()
    return d


def random_distribution_assignments(l, k, n, m):
    """sample n assignments from m random distributions proportionally based on similarity between each random
    distribution and equal distribution"""

    random_distributions = [generate_random_distribution(k) for _ in range(m)]
    equal_distribution = np.ones(k) / k

    # cosine similarity between random and equal distribution
    cos_sim = [dot(random_distribution, equal_distribution) / (norm(random_distribution) * norm(equal_distribution)) for
               random_distribution in random_distributions]
    n_distribution_sample = [int(i) for i in cos_sim[:-1] / sum(cos_sim) * n]
    n_distribution_sample.append(n - sum(n_distribution_sample))
    sampled_partitions = []

    for i, distribution in enumerate(random_distributions):
        n_sampled = 0
        while n_sampled < n_distribution_sample[i]:
            partition = random_distribution_assignment(l, k, distribution)
            if partition in sampled_partitions:
                continue
            else:
                sampled_partitions.append(partition)
                n_sampled += 1

    return sampled_partitions
