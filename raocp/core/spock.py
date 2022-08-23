import time
import raocp.core.cache as core_cache


class Spock:
    """
    Chambolle-Pock algorithm for RAOCPs accelerated by SuperMann
    """

    def __init__(self, cache: core_cache.Cache, max_outer_iters, max_inner_iters, c0, c1, c2,
                 beta, sigma, lamda, alpha):
        self.__cache = cache
        self.__max_outer_iters = max_outer_iters
        self.__max_inner_iters = max_inner_iters
        self.get_alphas()
        self.__counter_chock_operator = 0
        self.__andersons_setup_iterations = 3
        self.__initial_state = None

    def run(self, initial_state, c0=0.99, c1=0.99, c2=0.99, beta=0.5, sigma=0.1, lamda=1.95, alpha=0.5):
        """
        Chambolle-Pock algorithm accelerated by SuperMann

        zen is a column vector of primal stacked on top of dual
        """
        self.__initial_state = initial_state
        self.__cache.cache_initial_state(self.__initial_state)
        self.get_alphas()
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
            accepted_prim, accepted_dual = self.vector_to_parts(zen_k)
            self.__cache.set_primal(accepted_prim)
            self.__cache.set_dual(accepted_dual)
            self.__cache.update_cache()
            # step 1
            candidate_zen_kplus1 = self.get_new_zen(zen_k)
            resid_zen_k = self.get_residual(zen_k, candidate_zen_kplus1)
            norm_resid_zen_k = self.get_norm(resid_zen_k)
            # check termination criteria
            old_prim, old_dual = self.vector_to_parts(zen_k)
            new_prim, new_dual = self.vector_to_parts(candidate_zen_kplus1)
            current_error = self.get_current_error(new_prim, old_prim, new_dual, old_dual)
            self.cache_errors(i)
            if current_error <= self.__tol:
                break
            # initialise
            if i == 0:
                eta[pos_k] = norm_resid_zen_k
                r_safe = norm_resid_zen_k
            # step 2
            if i <= self.__andersons_setup_iterations:
                direction = self.andersons_setup(zen_k, resid_zen_k, i)
            else:
                if i != self.__andersons_setup_iterations + 1:
                    self.andersons_update_buffer(zen_k, resid_zen_k, i)
                direction = self.andersons_direction(i)
            # step 3
            if norm_resid_zen_k <= c0 * eta[pos_k]:
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
                norm_resid_w = self.get_norm(resid_w)
                # step 5a
                if norm_resid_zen_k <= r_safe and norm_resid_w <= c1 * norm_resid_zen_k:
                    zen_k = vector_w_k
                    r_safe = norm_resid_w + c2 ** i
                    counter_k1 += 1
                    break  # K_1
                # step 5b
                rho = norm_resid_w ** 2 - 2 * alpha * self.chock_inner_prod(resid_w, vector_w_k - zen_k)
                if rho >= sigma * norm_resid_w * norm_resid_zen_k:
                    zen_k = zen_k - lamda * (rho / norm_resid_w ** 2) * resid_w
                    counter_k2 += 1
                    break  # K_2
                else:
                    tau *= beta

        tock = time.perf_counter()
        print(f"timer stopped in {tock - tick:0.4f} seconds\n"
              f"k_0 = {counter_k0}, k_1 = {counter_k1}, k_2 = {counter_k2}")
        iter_ = i + 1
        return self.check_convergence(iter_), iter_, self.__counter_chock_operator

    def get_new_zen(self, vector_x_k_):
        prim_k_, dual_k_ = self.vector_to_parts(vector_x_k_)
        prim_kplus1_, dual_kplus1_ = self.chock_operator(prim_k_, dual_k_)
        vector_x_kplus1_ = self.parts_to_vector(prim_kplus1_, dual_kplus1_)
        return vector_x_kplus1_

    def get_new_w(self, vector_):
        # for speeding up new w by making chock operator a function of tau
        new_w_ = self.get_new_zen(vector_)
        return new_w_

    @staticmethod
    def get_residual(vector_k_, vector_kplus1_):
        residual = vector_k_ - vector_kplus1_
        return residual

    def get_initial_vector_zen(self):
        setup_prim_, setup_dual_ = self.__cache.get_primal_and_dual()
        setup_zen_ = self.parts_to_vector(setup_prim_, setup_dual_)
        return setup_zen_
