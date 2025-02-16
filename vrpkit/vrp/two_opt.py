from vrpkit.base.formulation import VRP
from vrpkit.base.solution import RoutePlan


def two_opt(solution: RoutePlan, vrp: VRP, max_iter=10):
    """Optimize a route plan by doing 2-opt heuristic
    Notice: it WILL change the original route plan!
    """

    for veh_id in solution:
        iter_num = 0
        improved = True
        origin_route = solution[veh_id]
        route_len = len(origin_route)
        best_route = origin_route

        best_obj = vrp.evaluate_route(veh_id, origin_route)

        while improved and iter_num <= max_iter:
            improved = False

            for i in range(1, route_len - 2):
                for j in range(i + 1, route_len):
                    new_route = best_route.copy()
                    new_route[i:j] = new_route[i:j][::-1]
                    new_obj = vrp.evaluate_route(veh_id, new_route)
                    if new_obj > best_obj:
                        best_route = new_route
                        best_obj = new_obj
                        improved = True

            iter_num += 1

        solution[veh_id] = best_route

    return solution
