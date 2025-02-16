from vrpkit.vrp.formulation import load_instance
from vrpkit.base.utils import plot_solution, test_solver
from vrpkit.vrp.hga import HGA


def test_heuristics(vrp):
    from vrpkit.vrp.hga import random_vehicle_route_split
    from vrpkit.vrp.two_opt import two_opt
    from vrpkit.vrp.nns import nearest_neighbour_search
    depot = vrp.asset.depots.values_in_list()[0]

    solution = random_vehicle_route_split(depot, vrp.asset, vrp.task.values_in_list())
    solution.display()

    nearest_neighbour_search(solution, vrp.cost_matrix)
    solution.display()

    two_opt(solution, vrp, 10)
    solution.display()


def test_hga(vrp):
    from vrpkit.vrp.hga import HGA
    solver = HGA()
    solver.solve(vrp)
    vrp.solution.display()
    fig, _ = plot_solution(vrp, vrp.solution)
    fig.show()


if __name__ == "__main__":
    from vrpkit.vrp.insertion import NearestInsertionHeuristic
    from vrpkit.vrp.hga import HGA

    nih_solver = NearestInsertionHeuristic()
    hga_solver = HGA()
    nih_solver.config(depot_assignment_method="approx",
                      do_post_2opt=True,
                      post_2opt_iter_n=1,
                      do_post_opt_split=True)
    instance_folder = "./data/instances/middle/C_100_4"
    vrp = sum(load_instance(instance_folder))
    vrp.objective.coefficients = {'Gross Revenue': 1,
                                  'Travel Time': -1,
                                  'Service Delay': 0,
                                  'Idle Time': 0,
                                  'Over Load': -1000}

    test_solver(vrp, nih_solver)
    hga_solver.initialize(vrp)
    hga_solver.feed(nih_solver.best_solution)
    test_solver(vrp, hga_solver, initialize=False)
