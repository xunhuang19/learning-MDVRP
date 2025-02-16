import copy
from functools import partial
import random
import numpy as np
from expt.learning_cvrp.solver import Solver
from expt.learning_cvrp.train import get_prepare_subproblem
from expt.learning_cvrp.utils import MDVRProblem, MDVRSubProblem, Namespace, pad_each, Allocation
from expt.utils import random_assignment
from deap import base, tools
import torch
from vrpkit.vrp.hga import cluster_depot_order_assign


def random_allocations(all_customers, n_depot):
    partitions = random_assignment(all_customers, n_depot)

    allocations = []
    for i in range(len(partitions)):
        allocations += partitions[i]
        allocations.append(',')
    return allocations[:len(allocations) - 1]


# Remove crossover slice duplicated in the individual
def remove_duplicate(allocation, crossover_slice):
    for i, sub_allocation in enumerate(allocation):
        for j, customer in enumerate(sub_allocation):
            if customer in crossover_slice:
                allocation[i][j] = '_'


def insert_missing(allocation, crossover_slice1, crossover_slice2):
    crossover_slice = [customer for customer in crossover_slice1 if customer not in crossover_slice2]
    occurrence = 0
    for i, sub_allocation in enumerate(allocation):
        for j, customer in enumerate(sub_allocation):
            if occurrence >= len(crossover_slice):
                break
            if customer == '_':
                allocation[i][j] = crossover_slice[occurrence]
                occurrence = occurrence + 1


def rm_redundant_add_missing(allocation, prev_flat_list):
    # remove redundant '_'
    allocation = [[customer for customer in sub_allocation if customer != '_'] for sub_allocation in allocation]

    # add missing elements
    after_flat_list = [customer for sub_allocation in allocation for customer in sub_allocation]
    for customer in prev_flat_list:
        if customer not in after_flat_list:
            allocation[-1].append(customer)
    return allocation


class Fitness(base.Fitness):
    weights = (1,)


class Individual(Allocation):
    def __init__(self):
        super(Individual, self).__init__()
        self.fitness = Fitness()

    # partition one single allocation to n list for transformer model input
    def partition_ind(self):
        allocation = self.allocation
        par_indices = [i for i, x in enumerate(allocation) if x == ',']
        partition = [[] for _ in range(len(par_indices) + 1)]

        for i, idx in enumerate(par_indices):
            partition[i] = allocation[0:idx] if i == 0 else allocation[par_indices[i - 1] + 1:idx]
        partition[-1] = allocation[par_indices[-1] + 1:]

        return partition

    def __repr__(self):
        return str(self.allocation)


class GA(Solver):
    def __init__(self, mdvrp: MDVRProblem = None):
        super(GA, self).__init__(mdvrp)
        self.population = None
        self.evaluate = None
        self.select = None
        self.statistics = None
        self.logbook = None
        self.tournsize = 3
        self.evolve_protect_rate = 0.5
        self.pop_size = 5
        self.mutate_pb = 0.7
        self.crossover_pb = 0.1
        self.gen_n = 10

    def initialize(self, vrps=None, mdvrp=None, model=None):
        super(GA, self).initialize(mdvrp)
        self.evaluate = partial(GA.evaluate, mdvrp=self.mdvrp, model=model, vrps=vrps)
        self.select = partial(tools.selTournament, tournsize=self.tournsize)
        self.statistics = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.statistics.register("min", min)
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + self.statistics.fields

        self.init_population(self.pop_size, vrps)

    def init_population(self, pop_size, vrps, cluster_pb=0.8):
        self.population = []
        vrp = sum(vrps)
        orders = vrp.task.values_in_list()
        depots = vrp.asset.depots.values_in_list()
        cost_matrix = vrp.cost_matrix
        total_comp_num = len(depots)
        assets = [vrp.asset for vrp in vrps]
        all_order_ids = vrp.task.keys_in_list()

        valid_n = 0
        # Generate individual from the nearest neighbourhood search if a probability is lower than cluster pb
        while valid_n < pop_size:
            assignment = []
            ind = Individual()
            if random.random() < cluster_pb:
                depot_orders = cluster_depot_order_assign(depots, orders, cost_matrix)
                for depot in depot_orders:
                    id = []
                    for order in depot:
                        id.append(order.id)
                    assignment.append(id)
            # otherwise use random assignment
            else:
                assignment = random_assignment(all_order_ids, total_comp_num)
            # Ensure capcity constraint is not violated
            for i in range(total_comp_num):
                depot_vehicles = assets[i].depot_vehicles()
                max_cap = sum([v.capacity for _, vehicles in depot_vehicles.items() for v in vehicles])
                order_vol = sum([vrp.task.data[order_id].volume for order_id in assignment[i]])
                if order_vol > max_cap:
                    break
                elif i == total_comp_num - 1:
                    valid_n += 1

            ind.allocation = assignment
            self.population.append(ind)

    def run(self, *args, **kwargs):
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
            offspring = self.select(population, self.pop_size)

            # clone individuals to remove dependency
            offspring = [GA.clone(ind) for ind in offspring]

            # crossover - varying population
            for i in range(1, len(offspring), 2):
                if random.random() < self.crossover_pb:
                    offspring[i - 1], offspring[i] = GA.crossover(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < self.mutate_pb:
                    offspring[i], = GA.swap_mutation(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # make sure that some best individual can always survive to the next iteration
            population.sort(key=lambda ind: ind.fitness.values, reverse=True)
            offspring.extend(population[-max(1, int(self.evolve_protect_rate * self.pop_size)):])

            # Replace the current population by the offspring
            population[:] = offspring
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        best_individual = min(population, key=lambda ind: ind.fitness.values)
        self.best_solution = best_individual

        return population

    @staticmethod
    def clone(offspring):
        offspring_clone = copy.deepcopy(offspring)
        return offspring_clone

    @staticmethod
    def evaluate(individual: Individual, mdvrp: MDVRProblem, model):
        """
        Evaluate the individual by the learned model
        """
        x = mdvrp.get_node_features()
        allocation = individual.allocation

        subp = MDVRSubProblem(mdvrp, allocation, None)
        node_idxs = subp.get_node_idx()

        # create a dummy subproblem to get the correct form for model input
        input_node_idxs = [node_idxs, np.zeros_like(node_idxs)]

        n_subp_nodes = np.array([len(ni) for ni in input_node_idxs])
        offsets = np.zeros_like(n_subp_nodes)
        data = Namespace(xs=x, offsets=offsets, node_idxs=pad_each(input_node_idxs), n_subp_nodes=n_subp_nodes)

        with torch.no_grad():
            obj_value = model(get_prepare_subproblem()(data)).cpu().numpy()[0]  # only output the first pred

        return obj_value,  # required by deap internal algorithm

    @staticmethod
    def swap_mutation(individual: Individual):
        allocation = individual.allocation
        loc_i1, loc_i2 = random.randint(0, len(allocation) - 1), random.randint(0, len(allocation) - 1)
        while loc_i1 == loc_i2:
            loc_i1, loc_i2 = random.randint(0, len(allocation) - 1), random.randint(0, len(allocation) - 1)
        if len(allocation[loc_i1]) == 0 or len(allocation[loc_i2]) == 0:
            return individual,
        else:
            loc_j1, loc_j2 = random.randint(0, len(allocation[loc_i1]) - 1), random.randint(0, len(allocation[loc_i2]) - 1)
            temp = allocation[loc_i1][loc_j1]
            allocation[loc_i1][loc_j1] = allocation[loc_i2][loc_j2]
            allocation[loc_i2][loc_j2] = temp
            return individual,

    @staticmethod
    def crossover(ind1: Individual, ind2: Individual):
        allocation1, allocation2 = ind1.allocation, ind2.allocation
        prev_flat_list = [customer for sub_allocation in allocation1 for customer in sub_allocation][:]
        loc = random.randint(0, len(allocation1) - 1)
        temp1 = allocation1[loc][:]
        temp2 = allocation2[loc][:]

        remove_duplicate(allocation1, temp2)
        remove_duplicate(allocation2, temp1)

        # exchange crossover slice
        allocation1[loc] = temp2[:]
        allocation2[loc] = temp1[:]

        insert_missing(allocation1, temp1, temp2)
        insert_missing(allocation2, temp2, temp1)

        allocation1 = rm_redundant_add_missing(allocation1, prev_flat_list)
        allocation2 = rm_redundant_add_missing(allocation2, prev_flat_list)

        ind1.allocation = allocation1
        ind2.allocation = allocation2
        return ind1, ind2
