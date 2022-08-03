import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
import time
import raocp.core.cache as cache
import raocp.core.operators as ops
import raocp.core.raocp_spec as spec

import matplotlib.pyplot as plt
import tikzplotlib as tikz


class Solver:
    """
    Solver for RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec: spec.RAOCP, tol=1e-5, max_iters=10):
        self.__raocp = problem_spec
        self.__cache = cache.Cache(self.__raocp)
        self.__operator = ops.Operator(self.__cache)
        self.__initial_state = None
        self.__parameter_1 = None
        self.__parameter_2 = None
        self.__chock_iter = 0
        self.__max_iters = max_iters
        self.__tol = tol
        self.__error = [np.zeros(1)] * 3
        self.__delta_error = [np.zeros(1)] * 3
        self.__error_cache = None
        self.__delta_error_cache = None
        self.__counter_chock_operator = 0
        # andersons direction
        self.__andys_memory_size = 3
        self.__andys_x = None
        self.__andys_g = None
        self.__andys_matrix_x_k = None
        self.__andys_matrix_g_k = None
        self.__andys_counter_k = None

    def primal_k_plus_half(self):
        # get memory space for ell_transpose_dual
        _, ell_transpose_dual = self.__cache.get_primal()
        # get current dual
        _, old_dual = self.__cache.get_dual()
        # operate L transpose on dual and store in ell_transpose_dual
        self.__operator.ell_transpose(old_dual, ell_transpose_dual)
        # get old primal
        _, old_primal = self.__cache.get_primal()
        # old primal minus (alpha1 times ell_transpose_dual)
        new_primal = [a_i - b_i for a_i, b_i in zip(old_primal, [j * self.__parameter_1
                                                                 for j in ell_transpose_dual])]
        self.__cache.set_primal(new_primal)

    def primal_k_plus_one(self):
        self.__cache.proximal_of_f(self.__parameter_1)

    def dual_k_plus_half(self):
        # get memory space for ell_transpose_dual
        _, ell_primal = self.__cache.get_dual()
        # get primal k+1 and k
        primal, old_primal = self.__cache.get_primal()
        # two times new primal minus old primal
        modified_primal = [a_i - b_i for a_i, b_i in zip([j * 2 for j in primal], old_primal)]
        # operate L on modified primal
        self.__operator.ell(modified_primal, ell_primal)
        # get old dual
        _, old_dual = self.__cache.get_dual()
        # old dual plus (gamma times ell_primal)
        new_dual = [a_i + b_i for a_i, b_i in zip(old_dual, [j * self.__parameter_2
                                                             for j in ell_primal])]
        self.__cache.set_dual(new_dual)

    def dual_k_plus_one(self):
        self.__cache.proximal_of_g_conjugate(self.__parameter_2)

    def calculate_chock_errors(self):
        # in this function, p = primal and d = dual
        p_new, p = self.__cache.get_primal()
        d_new, d = self.__cache.get_dual()

        # error 1
        p_minus_p_new = [a_i - b_i for a_i, b_i in zip(p, p_new)]
        p_minus_p_new_over_alpha1 = [a_i / self.__parameter_1 for a_i in p_minus_p_new]
        d_minus_d_new = [a_i - b_i for a_i, b_i in zip(d, d_new)]
        _, ell_transpose_d_minus_d_new = self.__cache.get_primal()  # get memory position
        self.__operator.ell_transpose(d_minus_d_new, ell_transpose_d_minus_d_new)
        xi_1 = [a_i - b_i for a_i, b_i in zip(p_minus_p_new_over_alpha1, ell_transpose_d_minus_d_new)]

        # error 2
        d_minus_d_new_over_alpha2 = [a_i / self.__parameter_2 for a_i in d_minus_d_new]
        p_new_minus_p = [a_i - b_i for a_i, b_i in zip(p_new, p)]
        _, ell_p_new_minus_p = self.__cache.get_dual()  # get memory position
        self.__operator.ell(p_new_minus_p, ell_p_new_minus_p)
        xi_2 = [a_i + b_i for a_i, b_i in zip(d_minus_d_new_over_alpha2, ell_p_new_minus_p)]

        # error 0
        _, ell_transpose_error2 = self.__cache.get_primal()  # get memory position
        self.__operator.ell_transpose(xi_2, ell_transpose_error2)
        xi_0 = [a_i + b_i for a_i, b_i in zip(xi_1, ell_transpose_error2)]

        # delta errors
        delta_1 = p_new_minus_p
        delta_2 = [a_i - b_i for a_i, b_i in zip(d_new, d)]
        _, ell_transpose_delta_2 = self.__cache.get_primal()
        self.__operator.ell_transpose(delta_2, ell_transpose_delta_2)
        delta_0 = [a_i - b_i for a_i, b_i in zip(delta_1, ell_transpose_delta_2)]

        return xi_0, xi_1, xi_2, delta_0, delta_1, delta_2

    def get_current_error(self):
        # calculate error
        xi_0, xi_1, xi_2, delta_0, delta_1, delta_2 = self.calculate_chock_errors()
        xi = [xi_0, xi_1, xi_2]
        delta = [delta_0, delta_1, delta_2]
        for i in range(3):
            inf_norm_xi = [np.linalg.norm(a_i, ord=np.inf) for a_i in xi[i]]
            inf_norm_delta = [np.linalg.norm(a_i, ord=np.inf) for a_i in delta[i]]
            self.__error[i] = np.linalg.norm(inf_norm_xi, np.inf)
            self.__delta_error[i] = np.linalg.norm(inf_norm_delta, np.inf)

        return max(self.__error)

    def get_alphas(self):
        # find alpha_1 and _2
        _, prim = self.__cache.get_primal()
        _, dual = self.__cache.get_dual()
        size_prim = np.vstack(prim).size
        size_dual = np.vstack(dual).size
        ell = LinearOperator(dtype=None, shape=(size_dual, size_prim),
                             matvec=self.__operator.linop_ell)
        ell_transpose = LinearOperator(dtype=None, shape=(size_prim, size_dual),
                                       matvec=self.__operator.linop_ell_transpose)
        ell_transpose_ell = ell_transpose * ell
        eigens, _ = eigs(ell_transpose_ell)
        ell_norm = np.real(max(eigens))
        one_over_norm = 0.999 / ell_norm
        self.__parameter_1 = one_over_norm
        self.__parameter_2 = one_over_norm

    def check_termination_criteria(self, current_error_, current_iteration_, keep_running_):
        # check stopping criteria
        if current_iteration_ >= self.__max_iters:
            keep_running_ = False
        if current_error_ <= self.__tol:
            keep_running_ = False
        if keep_running_ is True:
            current_iteration_ = current_iteration_ + 1
        return current_iteration_, keep_running_

    def check_convergence(self, current_iter_):
        if current_iter_ < self.__max_iters:
            return 0  # converged
        elif current_iter_ >= self.__max_iters:
            return 1  # not converged
        else:
            raise Exception("Iteration error in solver")

    def simple_chock(self, initial_state):
        """
        Chambolle-Pock algorithm, plain and simple
        """
        self.__initial_state = initial_state
        self.__cache.cache_initial_state(self.__initial_state)
        self.get_alphas()
        current_iteration = 0
        print("timer started")
        tick = time.perf_counter()
        keep_running = True
        while keep_running:
            self.chock_operator()

            # calculate current error
            current_error = self.get_current_error()

            # cache variables
            self.__cache.update_cache()
            # cache error
            if current_iteration == 0:
                self.__error_cache = np.array(self.__error)
                self.__delta_error_cache = np.array(self.__delta_error)
            else:
                self.__error_cache = np.vstack((self.__error_cache, np.array(self.__error)))
                self.__delta_error_cache = np.vstack((self.__delta_error_cache, np.array(self.__delta_error)))

            # check stopping criteria
            current_iteration, keep_running = self.check_termination_criteria(current_error,
                                                                              current_iteration,
                                                                              keep_running)

        tock = time.perf_counter()
        print(f"timer stopped in {tock - tick:0.4f} seconds")
        return self.check_convergence(current_iteration), self.__counter_chock_operator

    def super_chock(self, initial_state, c0=0.99, c1=0.99, c2=0.99, beta=0.5, sigma=0.1, lamda=1.95, alpha=0.5):
        """
        Chambolle-Pock algorithm accelerated by SuperMann
        """
        self.__initial_state = initial_state
        self.__cache.cache_initial_state(self.__initial_state)
        self.get_alphas()
        current_error = None
        accepted_x_kplus1 = None
        accepted = False
        pos_k = 0
        pos_kplus1 = 1
        print("timer started")
        tick = time.perf_counter()
        counter_sm = 0
        counter_sm_real = 0
        keep_running = True
        while keep_running:
            eta = [np.zeros(0)] * 2
            r_safe = None
            vector_x_k = self.setup_super_chock()
            repeat = True
            while repeat:
                norm_resid_x, candidate_vector_x_kplus1 = self.get_chock_norm_residual(vector_x_k, vector=True)
                if counter_sm == 0:
                    eta[pos_k] = norm_resid_x
                    r_safe = norm_resid_x
                if norm_resid_x < 1e-1:  # self.__tol:
                    repeat = False
                else:
                    update_direction = self.andersons_direction(vector_x_k, candidate_vector_x_kplus1)
                    if norm_resid_x <= c0 * eta[pos_k]:
                        eta[pos_kplus1] = norm_resid_x
                        accepted_x_kplus1 = vector_x_k + update_direction
                        accepted = True
                    else:
                        eta[pos_kplus1] = eta[pos_k]
                        tau = 1
                        vector_w_k = vector_x_k + tau * update_direction
                        norm_resid_w, resid_w = self.get_chock_norm_residual(vector_w_k, residual=True)
                        if norm_resid_x <= r_safe and norm_resid_w <= c1 * norm_resid_x:
                            accepted_x_kplus1 = vector_w_k
                            accepted = True
                            r_safe = norm_resid_w + c2**counter_sm
                        else:
                            rho = norm_resid_w**2 - 2 * alpha * self.chock_inner_prod(resid_w, vector_w_k - vector_x_k)
                            if rho >= sigma * norm_resid_w * norm_resid_x:
                                accepted_x_kplus1 = vector_x_k - lamda * (rho / norm_resid_w**2) * resid_w
                                accepted = True
                            else:
                                tau *= beta
                if not repeat:
                    current_error = self.get_current_error()
                elif accepted:
                    current_error = self.get_current_error()
                    vector_x_k = accepted_x_kplus1
                    accepted = False
                    counter_sm_real += 1
                if counter_sm == 0:
                    self.__error_cache = np.array(self.__error)
                else:
                    self.__error_cache = np.vstack((self.__error_cache, np.array(self.__error)))
                eta[pos_k] = eta[pos_kplus1]
                counter_sm += 1

            # check stopping criteria
            self.__chock_iter, keep_running = self.check_termination_criteria(current_error,
                                                                              self.__chock_iter,
                                                                              keep_running)

        tock = time.perf_counter()
        print(f"timer stopped in {tock - tick:0.4f} seconds")
        return self.check_convergence(self.__chock_iter), self.__chock_iter, counter_sm_real, self.__counter_chock_operator

    def get_chock_norm_residual(self, vector_x_k_, residual=False, vector=False):
        prim_k_, dual_k_ = self.vector_to_parts(vector_x_k_)
        self.__cache.set_primal(prim_k_)
        self.__cache.set_dual(dual_k_)
        self.__cache.update_cache()
        self.chock_operator()
        prim_kplus1_, _ = self.__cache.get_primal()
        dual_kplus1_, _ = self.__cache.get_dual()
        vector_x_kplus1_ = self.parts_to_vector(prim_kplus1_, dual_kplus1_)
        resid_ = self.chock_residual(vector_x_k_, vector_x_kplus1_)
        norm_ = self.chock_norm(resid_)
        if residual:
            return norm_, resid_
        elif vector:
            return norm_, vector_x_kplus1_
        else:
            return norm_

    def chock_operator(self):
        # run primal part of algorithm
        self.primal_k_plus_half()
        self.primal_k_plus_one()
        # run dual part of algorithm
        self.dual_k_plus_half()
        self.dual_k_plus_one()
        # increase counter
        self.__counter_chock_operator = self.__counter_chock_operator + 1

    def chock_norm(self, vector_):
        norm = np.sqrt(self.chock_inner_prod(vector_, vector_))
        return norm

    def chock_norm_squared(self, vector_):
        norm = self.chock_inner_prod(vector_, vector_)
        return norm

    def chock_inner_prod(self, vector_a_, vector_b_):
        if vector_a_.shape[1] != 1 or vector_b_.shape[1] != 1:
            raise Exception("non column vectors provided to inner product")
        inner = vector_a_.T @ self.chock_inner_prod_matrix(vector_b_)
        return inner[0]

    @staticmethod
    def chock_residual(vector_k_, vector_kplus1_):
        residual = vector_k_ - vector_kplus1_
        return residual

    def chock_inner_prod_matrix(self, vector_a_):
        prim_, dual_ = self.vector_to_parts(vector_a_)
        ell_transpose_dual, _ = self.__cache.get_primal()
        ell_prim, _ = self.__cache.get_dual()
        self.__operator.ell_transpose(dual_, ell_transpose_dual)
        self.__operator.ell(prim_, ell_prim)
        modified_prim = [a_i - (self.__parameter_1 * b_i) for a_i, b_i in zip(prim_, ell_transpose_dual)]
        modified_dual = [a_i - (self.__parameter_2 * b_i) for a_i, b_i in zip(dual_, ell_prim)]
        return self.parts_to_vector(modified_prim, modified_dual)

    def setup_super_chock(self):
        setup_prim_x_kminus1, _ = self.__cache.get_primal()
        setup_dual_x_kminus1, _ = self.__cache.get_dual()
        setup_vector_x_kminus1 = self.parts_to_vector(setup_prim_x_kminus1, setup_dual_x_kminus1)
        _, setup_vector_x_k = self.get_chock_norm_residual(setup_vector_x_kminus1, vector=True)
        _, setup_vector_x_kplus1 = self.get_chock_norm_residual(setup_vector_x_k, vector=True)
        _, setup_vector_x_kplus2 = self.get_chock_norm_residual(setup_vector_x_kplus1, vector=True)
        self.andersons_setup(setup_vector_x_k, setup_vector_x_kplus1, setup_vector_x_kplus2)
        return setup_vector_x_kplus2

    @staticmethod
    def parts_to_vector(prim_, dual_):
        return np.vstack((np.vstack(prim_), np.vstack(dual_)))

    def vector_to_parts(self, vector_):
        prim_, _ = self.__cache.get_primal()
        dual_, _ = self.__cache.get_dual()
        index = 0
        for i in range(len(prim_)):
            size_ = prim_[i].size
            prim_[i] = np.array(vector_[index: index + size_]).reshape(-1, 1)
            index += size_

        for i in range(len(dual_)):
            size_ = dual_[i].size
            dual_[i] = np.array(vector_[index: index + size_]).reshape(-1, 1)
            index += size_

        return prim_, dual_

    def andersons_setup(self, x_k_, x_kplus1_, x_kplus2_):
        self.__andys_x = [x_k_, x_kplus1_]  # list of iterates x
        self.__andys_g = [x_kplus1_ - x_k_, x_kplus2_ - x_kplus1_]  # list of residuals
        self.__andys_matrix_x_k = self.__andys_x[1] - self.__andys_x[0]  # matrix of increments in x
        self.__andys_matrix_g_k = self.__andys_g[1] - self.__andys_g[0]  # matrix of increments in residuals
        self.__andys_counter_k = 1
    
    def andersons_direction(self, old_x_kplus1_, old_x_kplus2_):
        m_k = min(self.__andys_counter_k, self.__andys_memory_size)
        self.__andys_x.append(old_x_kplus1_)
        self.__andys_g.append(old_x_kplus2_ - self.__andys_x[self.__andys_counter_k + 1])
        self.__andys_matrix_x_k = np.hstack((self.__andys_matrix_x_k,
                                             self.__andys_x[self.__andys_counter_k + 1] -
                                             self.__andys_x[self.__andys_counter_k]))
        self.__andys_matrix_g_k = np.hstack((self.__andys_matrix_g_k,
                                             self.__andys_g[self.__andys_counter_k + 1] -
                                             self.__andys_g[self.__andys_counter_k]))
        if self.__andys_matrix_x_k.shape[1] > m_k:
            self.__andys_matrix_x_k = np.delete(self.__andys_matrix_x_k, axis=1, obj=0)
            self.__andys_matrix_g_k = np.delete(self.__andys_matrix_g_k, axis=1, obj=0)
        self.__andys_counter_k = self.__andys_counter_k + 1

        # Least squares
        gamma_k = np.linalg.lstsq(self.__andys_matrix_g_k, self.__andys_g[self.__andys_counter_k], rcond=None)[0]

        direction = self.__andys_g[self.__andys_counter_k] - \
            ((self.__andys_matrix_x_k + self.__andys_matrix_g_k) @ gamma_k)

        return direction

    # print ###################################################
    def print_states(self):
        primal, _ = self.__cache.get_primal()
        seg_p = self.__cache.get_primal_segments()
        print(f"states =\n")
        for i in range(seg_p[1], seg_p[2]):
            print(f"{primal[i]}\n")

    def print_inputs(self):
        primal, _ = self.__cache.get_primal()
        seg_p = self.__cache.get_primal_segments()
        print(f"inputs =\n")
        for i in range(seg_p[2], seg_p[3]):
            print(f"{primal[i]}\n")

    def plot_residuals(self, solver):
        width = 2
        plt.semilogy(self.__error_cache[:, 0], linewidth=width, linestyle="solid")
        plt.semilogy(self.__error_cache[:, 1], linewidth=width, linestyle="solid")
        plt.semilogy(self.__error_cache[:, 2], linewidth=width, linestyle="solid")
        # plt.semilogy(self.__delta_error_cache[:, 0], linewidth=width, linestyle="dashed")
        # plt.semilogy(self.__delta_error_cache[:, 1], linewidth=width, linestyle="dashed")
        # plt.semilogy(self.__delta_error_cache[:, 2], linewidth=width, linestyle="dashed")
        plt.title(f"{solver} solver residual values of Chambolle-Pock algorithm iterations")
        plt.ylabel(r"log(residual value)", fontsize=12)
        plt.xlabel(r"iteration", fontsize=12)
        plt.legend(("xi_0", "xi_1", "xi_2"))  # , "delta_0", "delta_1", "delta_2"))
        tikz.save('4-3-residuals.tex')
        plt.show()

    @staticmethod
    def plot_residual_comparisons(xi, solver1, solver2, solver1_name, solver2_name):
        width = 2
        plt.semilogy(solver1.__error_cache[:, xi], linewidth=width, linestyle="solid")
        plt.semilogy(solver2.__error_cache[:, xi], linewidth=width, linestyle="solid")
        plt.title(f"comparison of {solver1_name} solver and {solver2_name} solver xi_{xi} residual value")
        plt.ylabel(r"log(residual value)", fontsize=12)
        plt.xlabel(r"iteration", fontsize=12)
        plt.legend((f"{solver1_name}", f"{solver2_name}"))
        plt.show()

    def plot_solution(self, solver):
        primal, _ = self.__cache.get_primal()
        seg_p = self.__cache.get_primal_segments()
        x = primal[seg_p[1]: seg_p[2]]
        u = primal[seg_p[2]: seg_p[3]]
        state_size = np.size(x[0])
        control_size = np.size(u[0])
        raocp = self.__cache.get_raocp()
        num_nonleaf = raocp.tree.num_nonleaf_nodes
        num_nodes = raocp.tree.num_nodes
        num_leaf = num_nodes - num_nonleaf
        num_stages = raocp.tree.num_stages
        fig, axs = plt.subplots(2, state_size, sharex="all", sharey="row")
        fig.set_size_inches(15, 8)
        fig.set_dpi(80)
        fig.suptitle(f"{solver} solver", fontsize=16)

        for element in range(state_size):
            for i in range(num_leaf):
                j = raocp.tree.nodes_at_stage(num_stages-1)[i]
                plotter = [[raocp.tree.stage_of(j), x[j][element][0]]]
                while raocp.tree.ancestor_of(j) >= 0:
                    anc_j = raocp.tree.ancestor_of(j)
                    plotter.append([[raocp.tree.stage_of(anc_j), x[anc_j][element][0]]])
                    j = anc_j

                x_plot = np.array(np.vstack(plotter))
                axs[0, element].plot(x_plot[:, 0], x_plot[:, 1])
                axs[0, element].set_title(f"state element, x_{element}(t)")

        for element in range(control_size):
            for i in range(num_leaf):
                j = raocp.tree.ancestor_of(raocp.tree.nodes_at_stage(num_stages-1)[i])
                plotter = [[raocp.tree.stage_of(j), u[j][element][0]]]
                while raocp.tree.ancestor_of(j) >= 0:
                    anc_j = raocp.tree.ancestor_of(j)
                    plotter.append([[raocp.tree.stage_of(anc_j), u[anc_j][element][0]]])
                    j = anc_j

                u_plot = np.array(np.vstack(plotter))
                axs[1, element].plot(u_plot[:, 0], u_plot[:, 1])
                axs[1, element].set_title(f"control element, u_{element}(t)")

        for ax in axs.flat:
            ax.set(xlabel='stage, t', ylabel='value')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        fig.tight_layout()
        tikz.save('python-solution.tex')
        plt.show()

