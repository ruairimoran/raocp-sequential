import time
import raocp.core.cache as core_cache


class Chock:
    """
    Chambolle-Pock algorithm for RAOCPs, plain and simple
    """
    def __init__(self, cache: core_cache.Cache, tol, max_iters):
        self.__cache = cache
        self.__max_iters = max_iters
        self.__tol = tol
        self.__cache.get_parameters()
        self.__counter_chock_operator = 0
        self.__initial_state = None

    def run(self, initial_state):
        self.__initial_state = initial_state
        self.__cache.cache_initial_state(self.__initial_state)
        i = 0
        print("timer started")
        tick = time.perf_counter()
        for i in range(self.__max_iters):
            old_prim, old_dual = self.__cache.get_primal_and_dual()
            new_prim, new_dual = self.__cache.chock_operator(old_prim, old_dual)
            self.__counter_chock_operator = self.__counter_chock_operator + 1

            # calculate and cache current error
            current_error = self.__cache.get_current_error(new_prim, old_prim, new_dual, old_dual)
            self.__cache.cache_errors(i)

            # cache variables
            self.__cache.update_cache()

            # check stopping criteria
            if current_error <= self.__tol:
                break

        tock = time.perf_counter()
        print(f"timer stopped in {tock - tick:0.4f} seconds")
        return i, self.__counter_chock_operator
