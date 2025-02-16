from .solution import RoutePlan
from .formulation import VRP

import warnings


class Solver:
    """Vehicle Routing Problem Solver"""

    def __init__(self, vrp: VRP = None):
        self._vrp = vrp if vrp is not None else VRP()
        self.best_solution = RoutePlan()
        self.is_initialized = False

    @property
    def vrp(self):
        return self._vrp

    @vrp.setter
    def vrp(self, vrp):
        if isinstance(vrp, VRP):
            self._vrp = vrp
            self.encode()

    def encode(self):
        """encode the VRP data model to the input data needed by the solver"""
        pass

    def config(self, para_dict: dict[str,] = None, **kwargs):
        """update parameters pertaining to the solver or the optimisation process"""
        all_para_dict = {**kwargs}
        if para_dict is not None:
            all_para_dict.update(para_dict)
        for var, value in all_para_dict.items():
            if hasattr(self, var):
                self.__setattr__(var, value)
            else:
                warnings.warn(f"solver {self.__class__} has no attribute {var}")

    def feed(self, *args, **kwargs):
        """feed the solver a or a set of solutions"""
        if not self.is_initialized:
            raise ValueError("The solver must be initialized before feeding solutions")

    def initialize(self, vrp=None, *args, **kwargs):
        """discard previously computed results and restart from scratch"""
        if vrp is not None:
            self.vrp = vrp
        self.is_initialized = True

    def run(self, *args, **kwargs):
        """start or continue the optimisation"""
        if not self.is_initialized:
            self.initialize()

    def decode(self):
        """decode the solution of solver to the self-defined solution"""
        pass

    def solve(self, vrp=None):
        self.initialize(vrp)
        self.run()
        self.decode()
        return self.best_solution
