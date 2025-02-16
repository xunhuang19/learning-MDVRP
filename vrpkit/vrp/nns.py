from vrpkit.base.constant import INF
from vrpkit.base.costmatrix import CostMatrix
from vrpkit.base.solution import RoutePlan, Route


def nearest_neighbour_search(solution: RoutePlan, cost_matrix: CostMatrix):
    """optimize a route plan by nearest neighbour search subject to travel cost
    Notice: it will change the original route plan
    """

    for k in solution:
        origin_order_seq = solution[k][1:-1]
        if len(origin_order_seq) == 0:
            continue

        depot = solution[k][0]
        new_order_seq = []
        # find the order nearest to the depot as the first order
        first_order = None
        nearest_cost = INF
        for order in origin_order_seq:
            cost = cost_matrix.cost(depot.node_id, order.node_id)
            if cost < nearest_cost:
                first_order = order
                nearest_cost = cost
        new_order_seq.append(first_order)

        # do the nearest neighbour search for other orders
        unsorted_orders = set(origin_order_seq) - {first_order}
        for _ in range(len(origin_order_seq) - 1):
            next_order = None
            next_nearest_cost = INF
            for order in unsorted_orders:
                cost = cost_matrix.cost(new_order_seq[-1].node_id, order.node_id)
                if cost < next_nearest_cost:
                    next_order = order
                    next_nearest_cost = cost
            new_order_seq.append(next_order)
            unsorted_orders -= {next_order}

        # update solution
        solution[k] = Route([depot] + new_order_seq + [depot])

    return solution
