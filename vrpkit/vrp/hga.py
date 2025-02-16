"""This module contains the Hybrid Generic Algorithm for Multi-depot Vehicle Routing"""
from vrpkit.base.constant import INF
from vrpkit.base.costmatrix import CostMatrix
from vrpkit.base.demand import Order
from vrpkit.base.facility import Depot, Asset
from vrpkit.base.solution import Route, RoutePlan, assemble_delivery_route
from vrpkit.base.formulation import VRP
from vrpkit.base.solver import Solver
from .nns import nearest_neighbour_search
from .two_opt import two_opt

import random
from itertools import accumulate
from functools import partial
from collections.abc import Sequence

from deap import base, tools


def random_chunking(sequence, n):
    """randomly split given sequence into n chunks"""
    seq_len = len(sequence)
    sequence_shuffled = random.sample(sequence, seq_len)
    chunk_sizes = [random.random() for _ in range(n)]
    cum_chunk_sizes = list(accumulate(chunk_sizes))
    factor = seq_len / cum_chunk_sizes[-1]
    cum_chunk_sizes = [int(size * factor + 0.5) for size in cum_chunk_sizes]
    chunk_indices = [0] + cum_chunk_sizes
    chunks = []
    for i, j in zip(chunk_indices[:-1], chunk_indices[1:]):
        chunks.append(sequence_shuffled[i:j])
    return chunks


def average(sequence):
    return sum(sequence) / len(sequence)


def random_depot_order_assign(depots: Sequence[Depot], orders: Sequence[Order]):
    """assign orders randomly to depots"""
    depot_orders = random_chunking(orders, len(depots))
    return depot_orders


def cluster_depot_order_assign(depots: Sequence[Depot], orders: Sequence[Order], cost_matrix: CostMatrix):
    """assign orders to its nearest depots"""
    depot_orders = [[] for _ in depots]
    for order in orders:
        nearest_depot_index = None
        nearest_cost = INF
        for i, depot in enumerate(depots):
            cost = cost_matrix.cost(depot.location.node_id, order.unloading_location.node_id)
            if cost < nearest_cost:
                nearest_depot_index = i
                nearest_cost = cost
        depot_orders[nearest_depot_index].append(order)
    return depot_orders


def random_vehicle_route_split(depot: Depot, asset: Asset, orders: Sequence[Order]):
    """split order list assigned to a depot into several vehicles routes randomly"""
    vehicles = asset.depot_vehicles(depot.id)[depot.id]
    route_orders = random_chunking(orders, len(vehicles))
    routes = RoutePlan()
    for i, vehicle in enumerate(vehicles):
        routes[vehicle.id] = assemble_delivery_route(depot, route_orders[i])
    return routes


def optimal_vehicle_route_split(depot: Depot, asset: Asset, orders: Sequence[Order], cost_matrix: CostMatrix):
    """split order list assigned to a depot into several vehicles routes by the Split Algorithm (Chu, 2006).

    This function reproduces the Split Algorithm in https://doi.org/10.1016/j.ejor.2004.08.017.
    It formulates an auxiliary directed graph where each edge represents a possible vehicle route.
    In this graph, there are 1 + len(orders) vertices. They are sequenced as V := [depot] + [orders] as
    the given input. One can only go from V[i] to V[i+n] where 0<=i<=|V|, 1<=n and 0<=i+n<=|V|.
    Assuming vertices are named from 0 to |V|, the distance between vertices i and j is as follows:

    if j > i + 1, d = cost_matrix[depot][i+1] + cost_matrix[i+1][i+2] + ... + cost_matrix[j-1][j]
                      + cost_matrix[j][depot]
    if j = i + 1, d(i,j) = cost_matrix[depot][j] + cost_matrix[j][depot]

    Subsequently, from source node V[0] to destination node V[|V|], each feasible path, a list of edges,
    is a route split. Bellman-ford algorithm is used to find the shortest path which is treated as the
    optimal split subject to the input order sequences and vehicles.As we know, Bellman-ford iterates the
    paths using from 0 edge to |V|-1 edges. But in this algorithm, the edges used cannot exceed the number
    of vehicles as each edge is a vehicle route.

    The origin algorithm only considers a homogeneous fleet. Therefore, we use average capacities to tackle
    heterogeneous fleet capacities. The capacity constraint is not strictly guaranteed.
    """
    vehicles = asset.depot_vehicles(depot.id)[depot.id]
    total_veh_num = len(vehicles)
    total_order_num = len(orders)
    total_vertices_num = total_order_num + 1

    total_vehicle_capacity = sum([v.capacity for v in vehicles])
    total_demand = sum([order.volume for order in orders])
    if total_demand > total_vehicle_capacity:
        vehicle_capacity_limit = total_demand / total_veh_num
    else:
        vehicle_capacity_limit = total_vehicle_capacity / total_veh_num

    vertices = [depot.location]
    vertices.extend([i.unloading_location for i in orders])
    vertices_labels = [0]
    vertices_labels.extend([INF] * total_order_num)
    vertices_labels_new = vertices_labels[:]

    path_trace = {}  # used for trace last node in the shortest path
    vehicle_used = 1  # also is the number of edges used
    stable = False

    while not stable and vehicle_used < total_veh_num:
        stable = True  # assume there is no change in this iteration

        for i in range(1, total_vertices_num):
            if vertices_labels[i - 1] == INF:
                continue

            vehicle_load = 0  #
            distance = 0  # init distance from vertex i-1 to vertex j
            for j in range(i, total_vertices_num):
                vehicle_load += orders[j - 1].volume
                if vehicle_load > vehicle_capacity_limit:
                    break

                # update the distance from vertex i-1 to vertex j
                if j == i:
                    distance += cost_matrix.cost(depot.location.node_id, vertices[j].node_id)
                    distance += cost_matrix.cost(vertices[j].node_id, depot.location.node_id)
                else:
                    distance -= cost_matrix.cost(vertices[j - 1].node_id, depot.location.node_id)
                    distance += cost_matrix.cost(vertices[j - 1].node_id, vertices[j].node_id)
                    distance += cost_matrix.cost(vertices[j].node_id, vertices[j].node_id)

                # do relaxation
                if vertices_labels[i - 1] + distance < vertices_labels[j]:
                    vertices_labels_new[j] = vertices_labels[i - 1] + distance
                    path_trace[j] = i - 1
                    stable = False

        vehicle_used += 1
        vertices_labels = vertices_labels_new[:]

    # there this is only one vehicle, no need to split
    if len(path_trace) == 0:
        path_trace = {total_vertices_num - 1: 0}

    # if vehicle capacity is not enough, last route will not be labelled
    # the following make all unlabelled vertices a route
    if path_trace.get(total_vertices_num - 1, None) is None:
        path_trace[total_vertices_num - 1] = max(path_trace.keys())

    # translate the shortest path to split routes
    routes = RoutePlan()
    route_indices = []
    route_end_index = total_vertices_num - 1
    route_start_index = route_end_index
    while route_start_index != 0:
        route_start_index = path_trace[route_end_index]
        route_indices.append((route_start_index, route_end_index))
        route_end_index = route_start_index

    for i, (start_index, end_index) in enumerate(route_indices[:total_veh_num]):
        route_orders = vertices[start_index + 1:end_index + 1]
        vehicle_route = Route([depot.location] + route_orders + [depot.location])
        routes[vehicles[i].id] = vehicle_route

    # if split routes exceed available vehicle numbers, attach rest orders to the last vehicle route
    for start_index, end_index in route_indices[total_veh_num:]:
        route_orders = vertices[start_index + 1:end_index + 1]
        orig_route = routes[vehicles[-1].id]
        routes[vehicles[-1].id] = Route(orig_route[:-1] + route_orders + [orig_route[-1]])

    return routes


class Fitness(base.Fitness):
    weights = (1,)


class Individual(RoutePlan):

    def __init__(self, data=None):
        super(Individual, self).__init__(data)
        self.fitness = Fitness()


class HGA(Solver):
    """Hybrid Genetic Algorithm"""

    def __init__(self, vrp: VRP = None):
        super(HGA, self).__init__(vrp)
        self.population = None
        self.educate = None
        self.evaluate = None
        self.select = None
        self.statistics = None
        self.logbook = None
        self.educate_max_iter = 50
        self.tournsize = 3
        self.evolve_protect_rate = 0.05
        self.pop_size = 100
        self.nns_pb = 0.3
        self.split_pb = 0.4
        self.mutate_pb = 0.3
        self.crossover_pb = 0.3
        self.educate_pb = 0.1
        self.gen_n = 10

    def initialize(self, vrp=None, *args, **kwargs):
        super(HGA, self).initialize(vrp)

        self.educate = partial(HGA.educate, vrp=self.vrp, max_iter=self.educate_max_iter)
        self.evaluate = partial(HGA.evaluate, vrp=self.vrp)
        self.select = partial(tools.selTournament, tournsize=self.tournsize)
        self.statistics = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.statistics.register("max", max)
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + self.statistics.fields

        self.init_population(self.pop_size, self.nns_pb, self.split_pb)

    def init_population(self, pop_size=50, nns_pb=0.3, split_pb=0.4):
        if self.vrp.is_empty():
            raise ValueError("Routing Form is absent. HGA needs to be (re)initialized!")

        self.population = []
        orders = self.vrp.task.values_in_list()
        depots = self.vrp.asset.depots.values_in_list()
        cost_matrix = self.vrp.cost_matrix

        for _ in range(pop_size):

            ind = Individual()
            rand_num = random.random()
            shuffled_orders = random.sample(orders, len(orders))

            if rand_num <= nns_pb:
                # generate individual by following nearest neighbour principle
                depot_orders = cluster_depot_order_assign(depots, shuffled_orders, cost_matrix)
                for i, depot in enumerate(depots):
                    routes = random_vehicle_route_split(depot, self.vrp.asset, depot_orders[i])
                    ind.update(routes)
                ind = nearest_neighbour_search(ind, cost_matrix)

            elif rand_num <= nns_pb + split_pb:
                # randomly generate depot route but split it by Split Algorithm
                depot_orders = random_depot_order_assign(depots, shuffled_orders)
                for i, depot in enumerate(depots):
                    routes = optimal_vehicle_route_split(depot, self.vrp.asset, depot_orders[i], cost_matrix)
                    ind.update(routes)

            else:
                # randomly generate depot route and randomly split it
                depot_orders = random_depot_order_assign(depots, shuffled_orders)
                for i, depot in enumerate(depots):
                    routes = random_vehicle_route_split(depot, self.vrp.asset, depot_orders[i])
                    ind.update(routes)

            self.population.append(ind)

        return self.population

    def feed(self, solutions, *args, **kwargs):
        super(HGA, self).feed()
        if isinstance(solutions, RoutePlan):
            solutions = [solutions]
        for solution in solutions:
            self.population.append(Individual(solution))

    def run(self, *args, **kwargs):
        super(HGA, self).run()

        if self.population is None:
            self.init_population()

        population = self.population
        logbook = self.logbook
        stats = self.statistics

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = map(self.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(population)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)

        # Begin the generational process
        for gen in range(1, self.gen_n + 1):
            # Select the next generation individuals
            offspring = self.select(population, len(population))

            # clone individuals to remove dependency
            offspring = [HGA.clone(ind) for ind in offspring]

            # crossover - varying population
            for i in range(1, len(offspring), 2):
                if random.random() < self.crossover_pb:
                    offspring[i - 1], offspring[i] = HGA.ordered_crossover(offspring[i - 1],
                                                                           offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            # mutate - varying population
            for i in range(len(offspring)):
                if random.random() < self.mutate_pb:
                    offspring[i], = HGA.mutate(offspring[i])
                    del offspring[i].fitness.values

            # educate offspring - improve their qualities by 2-opt
            for i in range(len(offspring)):
                if random.random() < self.educate_pb:
                    offspring[i] = self.educate(offspring[i])

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # make sure that some best individual can always survive to the next iteration
            population.sort(key=lambda ind: ind.fitness.values)
            offspring.extend(population[-max(1, int(self.evolve_protect_rate * self.pop_size)):])

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        best_individual = max(population, key=lambda ind: ind.fitness.values)
        self.vrp.solution = self.educate(best_individual)
        self.best_solution = best_individual

        return population

    @staticmethod
    def clone(individual: Individual):
        return individual.copy()

    @staticmethod
    def evaluate(individual: Individual, vrp: VRP):
        obj_value, _ = vrp.evaluate(individual)
        return obj_value,  # required by deap internal algorithm

    @staticmethod
    def mutate(individual: Individual):
        """Mutation Operator"""
        method = random.randint(0, 2)
        if method == 0:
            offspring = HGA.swap_mutation(individual)
        elif method == 1:
            offspring = HGA.insert_mutation(individual)
        else:
            offspring = HGA.reverse_mutation(individual)
        return offspring,  # required by deap internal algorithm

    @staticmethod
    def educate(individual: Individual, vrp: VRP, max_iter=10):
        return two_opt(individual, vrp, max_iter)

    @staticmethod
    def swap_mutation(individual: Individual):
        """Mutation method - Swap"""
        if individual.is_empty():
            return individual

        route1, route2 = random.choices(list(individual.values()), k=2)
        if len(route1) > 2 and len(route2) > 2:
            # the first and last element are depots
            loc1, loc2 = random.randint(1, len(route1) - 2), random.randint(1, len(route2) - 2)
            if route1 == route2:  # means modify one will also change two!
                temp = route1[loc1]
                route1[loc1] = route1[loc2]
                route1[loc2] = temp
            else:
                route1[loc1], route2[loc2] = route2[loc2], route1[loc1]
        return individual

    @staticmethod
    def insert_mutation(individual: Individual):
        """Mutation method - Insertion"""
        if individual.is_empty():
            return individual

        route1, route2 = random.choices(list(individual.values()), k=2)
        if len(route1) > 2:
            loc1 = random.randint(1, len(route1) - 2)
            loc2 = random.randint(1, max(1, len(route2) - 2))
            route2.insert(loc2, route1.pop(loc1))
        return individual

    @staticmethod
    def reverse_mutation(individual: Individual):
        """Mutation method - Reversion"""
        if individual.is_empty():
            return individual

        route = random.choice(list(individual.values()))
        route_length = len(route)
        if route_length > 2:
            loc1, loc2 = random.randint(1, route_length - 2), random.randint(1, route_length - 2)
            if loc1 >= loc2:
                loc1, loc2 = loc2, loc1
            route[loc1:loc2] = route[loc1:loc2][::-1]
        return individual

    @staticmethod
    def ordered_crossover(ind1: Individual, ind2: Individual):
        """Crossover Operator"""
        if ind1.is_empty() or ind2.is_empty():
            # empty VRP
            return ind1, ind2

        loc1, loc2 = random.randint(0, len(ind1) - 1), random.randint(0, len(ind2) - 1)
        veh1, veh2 = ind1.keys_in_list()[loc1], ind2.keys_in_list()[loc2]
        slice_len = max(1, min(len(ind1[veh1]) - 2, len(ind2[veh2]) - 2))
        route1_slice, route2_slice = ind1[veh1][1:slice_len], ind2[veh2][1:slice_len]

        overlap = []
        for order in route1_slice:
            if order in route2_slice:
                overlap.append(order)
        for order in overlap:
            route1_slice.remove(order)
            route2_slice.remove(order)

        for i in range(len(route1_slice)):
            ind1.replace(route2_slice[i], route1_slice[i])
            ind2.replace(route1_slice[i], route2_slice[i])

        ind1[veh1][:slice_len], ind2[veh2][:slice_len] = ind2[veh2][:slice_len], ind1[veh1][:slice_len]

        return ind1, ind2
