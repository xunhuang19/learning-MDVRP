import warnings
import numpy as np
import yaml
from vrpkit.vrp.formulation import load_data, process_instance, MDVRPTW

yaml_types = (str, int, float, bool, type(None))


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(path, obj):
    with open(path, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False, allow_unicode=True)


def pad_to(array, *lengths, **kwargs):
    lengths = lengths + array.shape[len(lengths):]
    return np.pad(array, [(0, n_pad - n) for n_pad, n in zip(lengths, array.shape)], **kwargs)


def pad_each(array, length=None):
    length = length or max(len(row) for row in array)
    return np.array([pad_to(row, length) for row in array])


# convert data to Namespace type
class Namespace(dict):
    def __init__(self, *args, **kwargs):
        kvs = dict()
        for a in args:
            if type(a) is str:
                kvs[a] = True
            else:  # a is a dictionary
                kvs.update(a)
        kvs.update(kwargs)
        self.update(kvs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            self.__getattribute__(key)

    def __setattr__(self, key, value):
        self[key] = value

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        return self

    def new(self, *args, **kwargs):
        return Namespace({**self, **Namespace(*args, **kwargs)})


class MDVRProblem:
    def __init__(self, xys, demands, capacities, n_depot):
        self.xys = xys
        self.n_depot = n_depot
        self.depots = xys[0:self.n_depot]
        self.customers = xys[self.n_depot:]
        self.demands = demands[self.n_depot:]  # customer demand
        self._demands = demands
        self.capacities = capacities
        self.n_customer = len(self.demands)

    def get_node_features(self):
        """
        get the node features for each subproblem from their node index
        """
        xys = self.depots - self.depots
        demands = self._demands[0:self.n_depot]
        allocations = demands
        for i in range(self.n_depot):
            xys = np.concatenate([xys, self.customers - self.depots[i]])  # xys: the distance from customer to depot
            demands = np.concatenate([demands, self.demands / self.capacities[i]])  # demands normalisation
            allocations = np.concatenate([allocations, ((np.ones(self.n_customer) + i) / self.n_depot).reshape(-1,
                                                                                                               1)])  # assign allocation of each customer to depot index
        features = [xys, demands, allocations]
        return np.hstack(features)

    @classmethod
    def transform_features(cls, x, t, rotate=False, flip=False, perturb_node=False, perturb_route=False):
        """
        transform feature to feed into the network
        """
        if not (rotate or flip or perturb_node or perturb_route): return x, t
        import torch

        n_batch, n_node, _ = x.shape

        thetas = 2 * np.pi * (torch.rand if rotate else torch.zeros)(n_batch, device=x.device, dtype=x.dtype)
        s, c = thetas.sin(), thetas.cos()

        rot = torch.stack([c, -s, s, c]).view(2, 2, n_batch)
        if flip: rot[1, :, torch.randint(0, 2, (n_batch,), device=x.device, dtype=torch.bool)] *= -1

        noise_node = torch.normal(0, perturb_node, (n_batch, n_node, 2), device=x.device,
                                  dtype=x.dtype) if perturb_node else torch.zeros((n_batch, n_node, 2), device=x.device,
                                                                                  dtype=x.dtype)
        noise_node[:, 0] = 0

        x = torch.cat([torch.einsum('ijb,bfj->bfi', rot, x[:, :, :2] + noise_node), x[:, :, 2:]], dim=2)
        if t is not None:
            max_n_routes = t.size(1)
            noise_route = torch.normal(0, perturb_route, (n_batch, max_n_routes, 2), device=x.device,
                                       dtype=x.dtype) if perturb_route else torch.zeros((n_batch, max_n_routes, 2),
                                                                                        device=x.device, dtype=x.dtype)
            t = torch.cat([
                *(torch.einsum('ijb,bfj->bfi', rot, t[:, :, i: i + 2] + noise_route) for i in [0, 2, 4]),
                t[:, :, 6:]
            ], dim=2)
        return x, t


class MDVRSubProblem(MDVRProblem):
    def __init__(self, mdvrp, allocation, dist):
        super().__init__(mdvrp.xys, mdvrp._demands, mdvrp.capacities, mdvrp.n_depot)
        self.allocation = allocation
        self.dist = dist
        self.node_idxs = []

    def get_node_idx(self):
        """
         get node idx for each allocation within a list of nodes
        """
        for i in range(self.n_depot):
            # node_idx_i = [x + self.n_customer * i for x in self.allocation[i]]
            node_idx_i = [self.n_depot + self.n_customer * i + x for x in self.allocation[i]]
            self.node_idxs = np.concatenate([self.node_idxs, node_idx_i])
        return self.node_idxs


class Allocation:
    def __init__(self):
        self.allocation = None


def get_mdvrp(instance_dir):
    carrier_data, depot_data, vehicle_data, cust_data = load_data(instance_dir)
    n_depot = len(depot_data)
    _, _, xys, demands, capacities = process_instance(depot_data, vehicle_data, cust_data)
    problem = MDVRProblem(xys, demands, capacities, n_depot)
    return problem


def get_cost(vrps, assignment, solver):
    central_vrp = sum(vrps)
    assets = [vrp.asset for vrp in vrps]  # each asset represent a logistics service provider
    total_comp_num = len(assets)
    tt = 0
    for i in range(total_comp_num):
        task = central_vrp.task.subset(assignment[i])
        if not task.is_empty():
            vrp = MDVRPTW(assets[i], task, central_vrp.cost_matrix, central_vrp.objective)
            solver.solve(vrp)
            _, metrics = vrp.evaluate()
            tt += metrics["Travel Time"]
    return tt


def get_solution(vrps, assignment, solver):
    """
    get local route plan for each carrier in a collaborative vehicle routing problem
    """
    central_vrp = sum(vrps)
    assets = [vrp.asset for vrp in vrps]  # each asset represent a logistics service provider
    total_comp_num = len(assets)
    tt = 0
    for i in range(total_comp_num):
        task = central_vrp.task.subset(assignment[i])
        if not task.is_empty():
            vrp = MDVRPTW(assets[i], task, central_vrp.cost_matrix, central_vrp.objective)
            solver.solve(vrp)
            _, metrics = vrp.evaluate()
            # vrp.solution.display()
            tt += metrics["Travel Time"]
            if metrics["Over Load"] != 0:
                warnings.warn(f"vehicle capacity constraint is violated")
    return tt
