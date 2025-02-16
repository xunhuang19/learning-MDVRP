from expt.learning_cvrp.utils import MDVRProblem, Allocation


class Solver:
    """Multi-depot Vehicle Routing Problem allocation Solver"""

    def __init__(self, mdvrp: MDVRProblem = None):
        self._mdvrp = mdvrp if mdvrp is not None else ValueError("MDVRP is absent. MDVRP needs to be initialised!")
        self.best_solution = Allocation()
        self.is_initialized = False

    @property
    def mdvrp(self):
        return self._mdvrp

    @mdvrp.setter
    def mdvrp(self, mdvrp):
        if isinstance(mdvrp, MDVRProblem):
            self._mdvrp = mdvrp

    def initialize(self, mdvrp=None, *args, **kwargs):
        """discard previously computed results and restart from scratch"""
        if mdvrp is not None:
            self.mdvrp = mdvrp
        self.is_initialized = True

    def run(self, *args, **kwargs):
        """start or continue the optimisation"""
        if not self.is_initialized:
            self.initialize()

    def solve(self, mdvrp=None):
        self.initialize(mdvrp)
        self.run()
        self.decode()
        return self.best_solution
