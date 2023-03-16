import raocp.core.cache as cache
import raocp.core.operators as ops
import raocp.core.raocp_spec as spec
import raocp.core.chock as core_chock
import raocp.core.spock as core_spock
import raocp.core.directions as core_directions


class Solver:
    """
    Solver for RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec: spec.RAOCP, tol=1e-3, max_iters=5000):
        self.__raocp = problem_spec
        self.__cache = cache.Cache(self.__raocp)
        self.__operator = ops.Operator(self.__cache)
        self.__cache.set_linear_operator(self.__operator)
        self.__tol = tol
        self.__max_iters = max_iters
        self.__max_outer_iters = None
        self.__solver = None
        self.__direction = None
        self.__do_supermann = False

    # BUILDERS #########################################################################################################

    def with_residuals_direction(self):
        self.__direction = core_directions.Residuals()
        return self

    def with_andersons_direction(self, memory_size=5):
        self.__direction = core_directions.Andersons(memory_size)
        return self

    def with_spock(self, max_outer_iters=5000, max_inner_iters=50, c0=0.99, c1=0.99, c2=0.99, 
                   beta=0.5, sigma=0.1, lamda=1.95, alpha=0.5):
        if self.__direction is None:
            raise Exception("'with_..._direction' must be given before 'with_spock")
        self.__solver = core_spock.Spock(self.__cache, self.__direction, self.__tol,
                                         max_outer_iters, max_inner_iters,
                                         c0, c1, c2, beta, sigma, lamda, alpha)
        self.__do_supermann = True
        self.__max_outer_iters = max_outer_iters
        return self

    # GETTERS ##########################################################################################################
        
    @property
    def get_cache(self):
        return self.__cache

    # RUN ##############################################################################################################
    
    def run(self, initial_state):
        if self.__do_supermann:
            iters, chock_op_calls = self.__solver.run(initial_state)
            status = 0 if iters < self.__max_outer_iters else 1
        else:
            self.__solver = core_chock.Chock(self.__cache, self.__tol, self.__max_iters)
            iters, chock_op_calls = self.__solver.run(initial_state)
            status = 0 if iters < self.__max_iters else 1
        return status, iters, chock_op_calls
