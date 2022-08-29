import time
import numpy as np
import raocp.core.cache as core_cache
import raocp.core.operators as core_operators


class Spock:
    """
    Chambolle-Pock algorithm for RAOCPs accelerated by SuperMann
    """

    def __init__(self, cache: core_cache.Cache, direction, tol, max_outer_iters, max_inner_iters,
                 c0, c1, c2, beta, sigma, lamda, alpha):
        self.__cache = cache
        self.__direction = direction
        self.__max_outer_iters = max_outer_iters
        self.__max_inner_iters = max_inner_iters
        self.__c0 = c0
        self.__c1 = c1
        self.__c2 = c2
        self.__beta = beta
        self.__sigma = sigma
        self.__lamda = lamda
        self.__alpha = alpha
        self.__tol = tol
        self.__cache.get_parameters()
        self.__counter_chock_operator = 0
        self.__andersons_setup_iterations = 3
        self.__initial_state = None

    def run(self, initial_state):
        """
        Chambolle-Pock algorithm accelerated by SuperMann

        zen is a column vector of primal stacked on top of dual
        """
        self.__initial_state = initial_state
        self.__cache.cache_initial_state(self.__initial_state)
        pos_k = 0
        pos_kplus1 = 1
        print("timer started")
        tick = time.perf_counter()
        eta = [np.zeros(0)] * 2
        r_safe = None
        zen_k = self.get_initial_vector_zen()
        i = 0
        counter_k0 = 0
        counter_k1 = 0
        counter_k2 = 0
        for i in range(self.__max_outer_iters):
            # step 6
            eta[pos_k] = eta[pos_kplus1]
            accepted_prim, accepted_dual = self.__cache.vector_to_parts(zen_k)
            self.__cache.set_primal(accepted_prim)
            self.__cache.set_dual(accepted_dual)
            self.__cache.update_cache()
            # step 1
            candidate_zen_kplus1, ell_prim_kplus1, ell_t_dual_kplus1 = self.get_new_zen(zen_k)
            resid_zen_k = self.get_residual(zen_k, candidate_zen_kplus1)
            norm_resid_zen_k = self.__cache.get_chock_norm(resid_zen_k)
            # check termination criteria
            old_prim, old_dual = self.__cache.vector_to_parts(zen_k)
            new_prim, new_dual = self.__cache.vector_to_parts(candidate_zen_kplus1)
            current_error = self.__cache.get_current_error(new_prim, old_prim, new_dual, old_dual,
                                                           ell_prim_kplus1, ell_t_dual_kplus1)
            self.__cache.cache_errors(i)
            if current_error <= self.__tol:
                break
            # initialise
            if i == 0:
                eta[pos_k] = norm_resid_zen_k
                r_safe = norm_resid_zen_k
            # step 2
            if self.__direction.is_residuals:
                direction = self.__direction.get_direction(resid_zen_k)
            elif self.__direction.is_andersons:
                if i <= self.__andersons_setup_iterations:
                    direction = self.__direction.run_setup(zen_k, resid_zen_k, i)
                else:
                    if i != self.__andersons_setup_iterations + 1:
                        self.__direction.update_buffer(zen_k, resid_zen_k, i)
                    direction = self.__direction.get_direction(i)
            else:
                raise Exception("Spock direction error")
            # step 3
            if norm_resid_zen_k <= self.__c0 * eta[pos_k]:
                eta[pos_kplus1] = norm_resid_zen_k
                zen_k = zen_k + direction
                counter_k0 += 1
                continue  # K_0
            # step 4
            eta[pos_kplus1] = eta[pos_k]
            tau = 1
            # step 5
            for j in range(self.__max_inner_iters):
                vector_w_k = zen_k + tau * direction
                candidate_w_kplus1 = self.get_new_w(vector_w_k)
                resid_w = self.get_residual(vector_w_k, candidate_w_kplus1)
                norm_resid_w = self.__cache.get_chock_norm(resid_w)
                # step 5a
                if norm_resid_zen_k <= r_safe and norm_resid_w <= self.__c1 * norm_resid_zen_k:
                    zen_k = vector_w_k
                    r_safe = norm_resid_w + self.__c2 ** i
                    counter_k1 += 1
                    break  # K_1
                # step 5b
                rho = norm_resid_w ** 2 - 2 * self.__alpha \
                    * self.__cache.get_chock_inner_prod(resid_w, vector_w_k - zen_k)
                if rho >= self.__sigma * norm_resid_w * norm_resid_zen_k:
                    zen_k = zen_k - self.__lamda * (rho / norm_resid_w ** 2) * resid_w
                    counter_k2 += 1
                    break  # K_2
                else:
                    tau *= self.__beta

        tock = time.perf_counter()
        print(f"timer stopped in {tock - tick:0.4f} seconds\n"
              f"k_0 = {counter_k0}, k_1 = {counter_k1}, k_2 = {counter_k2}")
        iter_ = i + 1
        return iter_, self.__counter_chock_operator

    def get_initial_vector_zen(self):
        setup_prim_, setup_dual_ = self.__cache.get_primal_and_dual()
        setup_zen_ = self.__cache.parts_to_vector(setup_prim_, setup_dual_)
        return setup_zen_

    def get_new_zen(self, vector_x_k_):
        prim_k_, dual_k_ = self.__cache.vector_to_parts(vector_x_k_)
        prim_kplus1_, dual_kplus1_, ell_prim_, ell_t_dual_ = self.__cache.chock_operator(prim_k_, dual_k_)
        self.__counter_chock_operator = self.__counter_chock_operator + 1
        vector_x_kplus1_ = self.__cache.parts_to_vector(prim_kplus1_, dual_kplus1_)
        return vector_x_kplus1_, ell_prim_, ell_t_dual_

    def get_new_w(self, vector_):
        # for speeding up new w by making chock operator a function of tau
        new_w_, _, _ = self.get_new_zen(vector_)
        return new_w_

    @staticmethod
    def get_residual(vector_k_, vector_kplus1_):
        residual = vector_k_ - vector_kplus1_
        return residual
