import random
import time
from itertools import chain, combinations

from matplotlib import pyplot as plt


def get_n_ramdom_colors(n):
    """randomly get n different colors"""
    random.seed(100)
    hexadecimal_alphabets = "0123456789ABCDEF"
    colors = ["#" + "".join([random.choice(hexadecimal_alphabets) for _ in range(6)]) for _ in range(n)]
    return colors


def powerset(iterable, n):
    """return all subset (size>=n) of an iterable"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(n, len(s) + 1))


def check_vrp_data(vrp):
    print(f"depots: {vrp.asset.depots}")
    print(f"vehicles: {vrp.asset.fleet}")
    print(f"orders: {vrp.task}")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    vrp.visualize(ax)
    plt.show()


def plot_solution(vrp, solution=None,
                  c_path="blue", arrow_width=0.02,
                  l_depo="Depot", c_depo="orange",
                  l_cust="Customer", c_cust_ulp="green", c_cust_lp="red",
                  legend=True, annotate=True):
    fig = plt.figure()
    ax = fig.add_subplot()
    vrp.visualize(ax, solution,
                  c_path, arrow_width,
                  l_depo, c_depo,
                  l_cust, c_cust_ulp, c_cust_lp,
                  legend, annotate)
    return fig, ax


def stirling(n, k):
    """calculate stirling number of the second kind"""
    if n <= 0:
        return 1
    elif k <= 0:
        return 0
    elif (n == 0 and k == 0):
        raise ValueError("n and k cannot be zero at the same time")
    elif n != 0 and n == k:
        return 1
    elif n < k:
        return 0
    else:
        return (k * (stirling(n - 1, k))) + stirling(n - 1, k - 1)


def stirling_eq(n, k):
    def accumulate_mul(sequence):
        res = 1
        for v in sequence:
            res = res * v
        return res

    total = 0
    for i in range(0, k + 1):
        a = (-1) ** (k - i)
        b = i ** n
        c = accumulate_mul(list(range(1, i + 1)))
        d = accumulate_mul(list(range(1, k - i + 1)))
        total += a * b / c / d
    return int(total)


def test_solver(vrp, solver, initialize=True):
    start_time = time.time()
    if initialize:
        solver.solve(vrp)
    else:
        solver.run()
        solver.decode()
    end_time = time.time()
    print(f"Solving Time: {end_time - start_time}")
    obj, kpis = vrp.evaluate()
    print(f"The objective value is {obj}.")
    print(f"The values of key performance metrics are {kpis}.")
    print("The solved route plan: ")
    vrp.solution.display()
    fig, _ = plot_solution(vrp, vrp.solution)
    fig.show()
