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

    def __init__(self, problem_spec: spec.RAOCP, tol=1e-3, max_outer_iters=1000, max_inner_iters=50):
        self.__raocp = problem_spec
        self.__cache = cache.Cache(self.__raocp)
        self.__operator = ops.Operator(self.__cache)
        self.__initial_state = None
        self.__parameter_1 = None
        self.__parameter_2 = None
        self.__store_operated_zen = None
        self.__store_operated_direction = None
        self.__max_outer_iters = max_outer_iters
        self.__max_inner_iters = max_inner_iters
        self.__andersons_setup_iterations = None
        self.__tol = tol
        self.__error = [np.zeros(1)] * 3
        self.__delta_error = [np.zeros(1)] * 3
        self.__error_cache = None
        self.__delta_error_cache = None
        self.__counter_chock_operator = 0
        # andersons direction
        self.__andys_memory_size = 10
        self.__andys_point = []
        self.__andys_resid = []
        self.__andys_point_diff = []
        self.__andys_resid_diff = []
        self.__andys_point_diff_matrix = None
        self.__andys_resid_diff_matrix = None

    def primal_k_plus_half(self, current_prim_, current_dual_):
        # get memory space for ell_transpose_dual
        ell_transpose_dual = self.__cache.get_primal()
        # operate L transpose on dual and store in ell_transpose_dual
        self.__operator.ell_transpose(current_dual_, ell_transpose_dual)
        # old primal minus (alpha1 times ell_transpose_dual)
        new_primal = [a_i - b_i for a_i, b_i in zip(current_prim_, [j * self.__parameter_1
                                                                    for j in ell_transpose_dual])]
        self.__cache.set_primal(new_primal)

    def primal_k_plus_one(self):
        self.__cache.proximal_of_primal(self.__parameter_1)

    def dual_k_plus_half(self, new_prim_, old_prim_, current_dual_):
        # get memory space for ell_transpose_dual
        ell_primal = self.__cache.get_dual()
        # two times new primal minus old primal
        modified_primal = [a_i - b_i for a_i, b_i in zip([j * 2 for j in new_prim_], old_prim_)]
        # operate L on modified primal
        self.__operator.ell(modified_primal, ell_primal)
        # old dual plus (gamma times ell_primal)
        new_dual = [a_i + b_i for a_i, b_i in zip(current_dual_, [j * self.__parameter_2
                                                                  for j in ell_primal])]
        self.__cache.set_dual(new_dual)

    def dual_k_plus_one(self):
        self.__cache.proximal_of_dual(self.__parameter_2)

    def calculate_chock_errors(self, new_prim_, old_prim_, new_dual_, old_dual_):
        # in this function, p = primal and d = dual
        p_new = new_prim_
        p = old_prim_
        d_new = new_dual_
        d = old_dual_

        # error 1
        p_minus_p_new = [a_i - b_i for a_i, b_i in zip(p, p_new)]
        p_minus_p_new_over_alpha1 = [a_i / self.__parameter_1 for a_i in p_minus_p_new]
        d_minus_d_new = [a_i - b_i for a_i, b_i in zip(d, d_new)]
        ell_transpose_d_minus_d_new = self.__cache.get_primal()  # get memory position
        self.__operator.ell_transpose(d_minus_d_new, ell_transpose_d_minus_d_new)
        xi_1 = [a_i - b_i for a_i, b_i in zip(p_minus_p_new_over_alpha1, ell_transpose_d_minus_d_new)]

        # error 2
        d_minus_d_new_over_alpha2 = [a_i / self.__parameter_2 for a_i in d_minus_d_new]
        p_new_minus_p = [a_i - b_i for a_i, b_i in zip(p_new, p)]
        ell_p_new_minus_p = self.__cache.get_dual()  # get memory position
        self.__operator.ell(p_new_minus_p, ell_p_new_minus_p)
        xi_2 = [a_i + b_i for a_i, b_i in zip(d_minus_d_new_over_alpha2, ell_p_new_minus_p)]

        # error 0
        ell_transpose_error2 = self.__cache.get_primal()  # get memory position
        self.__operator.ell_transpose(xi_2, ell_transpose_error2)
        xi_0 = [a_i + b_i for a_i, b_i in zip(xi_1, ell_transpose_error2)]

        return xi_0, xi_1, xi_2

    def get_current_error(self, new_prim_, old_prim_, new_dual_, old_dual_):
        # calculate error
        xi_0, xi_1, xi_2 = self.calculate_chock_errors(new_prim_, old_prim_, new_dual_, old_dual_)
        xi = [xi_0, xi_1, xi_2]
        for i in range(3):
            inf_norm_xi = [np.linalg.norm(a_i, ord=np.inf) for a_i in xi[i]]
            self.__error[i] = np.linalg.norm(inf_norm_xi, np.inf)

        return max(self.__error)

    def cache_errors(self, i_):
        if i_ == 0:
            self.__error_cache = np.array(self.__error)
        else:
            self.__error_cache = np.vstack((self.__error_cache, np.array(self.__error)))

    def get_alphas(self):
        # find alpha_1 and _2
        prim, dual = self.__cache.get_primal_and_dual()
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

    def check_convergence(self, current_iter_):
        if current_iter_ < self.__max_outer_iters:
            return 0  # converged
        elif current_iter_ >= self.__max_outer_iters:
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
        for i in range(self.__max_outer_iters):
            old_prim, old_dual = self.__cache.get_primal_and_dual()
            new_prim, new_dual = self.chock_operator(old_prim, old_dual)

            # calculate and cache current error
            current_error = self.get_current_error(new_prim, old_prim, new_dual, old_dual)
            self.cache_errors(i)

            # cache variables
            self.__cache.update_cache()

            # check stopping criteria
            if current_error <= self.__tol:
                break

        tock = time.perf_counter()
        print(f"timer stopped in {tock - tick:0.4f} seconds")
        return self.check_convergence(current_iteration), self.__counter_chock_operator

    def super_chock(self, initial_state, c0=0.99, c1=0.99, c2=0.99, beta=0.5, sigma=0.1, lamda=1.95, alpha=0.5,
                    andersons_setup_iterations=3):
        """
        Chambolle-Pock algorithm accelerated by SuperMann

        zen is a column vector of primal stacked on top of dual
        """
        self.__initial_state = initial_state
        self.__cache.cache_initial_state(self.__initial_state)
        self.get_alphas()
        if andersons_setup_iterations >= 3:
            self.__andersons_setup_iterations = andersons_setup_iterations
        else:
            raise Exception("Number of andersons setup iterations must be >= 3")
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

    def chock_operator(self, current_prim_, current_dual_):
        # run primal part of algorithm
        self.primal_k_plus_half(current_prim_, current_dual_)
        self.primal_k_plus_one()
        # run dual part of algorithm
        mod_prim_ = self.__cache.get_primal()
        self.dual_k_plus_half(mod_prim_, current_prim_, current_dual_)
        self.dual_k_plus_one()
        # get new parts
        new_prim_, new_dual_ = self.__cache.get_primal_and_dual()
        # increase counter
        self.__counter_chock_operator = self.__counter_chock_operator + 1
        return new_prim_, new_dual_

    def get_new_zen(self, vector_x_k_):
        prim_k_, dual_k_ = self.vector_to_parts(vector_x_k_)
        prim_kplus1_, dual_kplus1_ = self.chock_operator(prim_k_, dual_k_)
        vector_x_kplus1_ = self.parts_to_vector(prim_kplus1_, dual_kplus1_)
        return vector_x_kplus1_

    def get_new_w(self, vector_):
        # for speeding up new w by making chock operator a function of tau
        new_w_ = self.get_new_zen(vector_)
        return new_w_

    def get_norm(self, vector_):
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
    def get_residual(vector_k_, vector_kplus1_):
        residual = vector_k_ - vector_kplus1_
        return residual

    def chock_inner_prod_matrix(self, vector_a_):
        prim_, dual_ = self.vector_to_parts(vector_a_)
        ell_transpose_dual, ell_prim = self.__cache.get_primal_and_dual()
        self.__operator.ell_transpose(dual_, ell_transpose_dual)
        self.__operator.ell(prim_, ell_prim)
        modified_prim = [a_i - (self.__parameter_1 * b_i) for a_i, b_i in zip(prim_, ell_transpose_dual)]
        modified_dual = [a_i - (self.__parameter_2 * b_i) for a_i, b_i in zip(dual_, ell_prim)]
        return self.parts_to_vector(modified_prim, modified_dual)

    def get_initial_vector_zen(self):
        setup_prim_, setup_dual_ = self.__cache.get_primal_and_dual()
        setup_zen_ = self.parts_to_vector(setup_prim_, setup_dual_)
        return setup_zen_

    @staticmethod
    def parts_to_vector(prim_, dual_):
        return np.vstack((np.vstack(prim_), np.vstack(dual_)))

    def vector_to_parts(self, vector_):
        prim_, dual_ = self.__cache.get_primal_and_dual()
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

    def andersons_setup(self, zen_kplus1_, resid_zen_kplus1_, iplus1_):
        i_ = iplus1_ - 1
        self.__andys_point.append(zen_kplus1_)  # list of iterates zen
        self.__andys_resid.append(resid_zen_kplus1_)  # list of residuals
        if len(self.__andys_resid) >= 2:
            self.__andys_point_diff.append(self.__andys_point[iplus1_] - self.__andys_point[i_])
            self.__andys_resid_diff.append(self.__andys_resid[iplus1_] - self.__andys_resid[i_])
            self.__andys_point_diff_matrix = self.__andys_point_diff[i_]  # matrix of increments in x
            self.__andys_resid_diff_matrix = self.__andys_resid_diff[i_]  # matrix of increments in residuals
        # return -residual
        direction_ = -resid_zen_kplus1_
        return direction_
    
    def andersons_direction(self, iplus1_):
        i_ = iplus1_ - 1
        # least squares
        gamma_k_ = np.linalg.lstsq(self.__andys_resid_diff_matrix, self.__andys_resid[i_], rcond=None)[0]
        direction_ = - self.__andys_resid[i_] \
                     - ((self.__andys_point_diff_matrix - self.__andys_resid_diff_matrix) @ gamma_k_)
        return direction_

    def andersons_update_buffer(self, zen_kplus1_, resid_zen_kplus1_, iplus2_):
        iplus1_ = iplus2_ - 1
        i_ = iplus1_ - 1
        m_k = min(i_, self.__andys_memory_size)
        self.__andys_point.append(zen_kplus1_)
        self.__andys_resid.append(resid_zen_kplus1_)
        self.__andys_point_diff.append(self.__andys_point[iplus1_] - self.__andys_point[i_])
        self.__andys_resid_diff.append(self.__andys_resid[iplus1_] - self.__andys_resid[i_])
        self.__andys_point_diff_matrix = np.hstack((self.__andys_point_diff_matrix, self.__andys_point_diff[i_]))
        self.__andys_resid_diff_matrix = np.hstack((self.__andys_resid_diff_matrix, self.__andys_resid_diff[i_]))
        if self.__andys_point_diff_matrix.shape[1] > m_k:
            self.__andys_point_diff_matrix = np.delete(self.__andys_point_diff_matrix, axis=1, obj=0)
            self.__andys_resid_diff_matrix = np.delete(self.__andys_resid_diff_matrix, axis=1, obj=0)

    # print ###################################################
    def print_states(self):
        primal = self.__cache.get_primal()
        seg_p = self.__cache.get_primal_segments()
        print(f"states =\n")
        for i in range(seg_p[1], seg_p[2]):
            print(f"{primal[i]}\n")

    def print_inputs(self):
        primal = self.__cache.get_primal()
        seg_p = self.__cache.get_primal_segments()
        print(f"inputs =\n")
        for i in range(seg_p[2], seg_p[3]):
            print(f"{primal[i]}\n")

    def plot_residuals(self, solver):
        width = 2
        plt.semilogy(self.__error_cache[:, 0], linewidth=width, linestyle="solid")
        plt.semilogy(self.__error_cache[:, 1], linewidth=width, linestyle="solid")
        plt.semilogy(self.__error_cache[:, 2], linewidth=width, linestyle="solid")
        plt.title(f"{solver} solver residual values of Chambolle-Pock algorithm iterations")
        plt.ylabel(r"log(residual value)", fontsize=12)
        plt.xlabel(r"iteration", fontsize=12)
        plt.legend(("xi_0", "xi_1", "xi_2"))
        tikz.save('4-3-residuals.tex')
        plt.show()

    @staticmethod
    def plot_residual_comparisons(xi, solver1, solver2, solver3, solver1_name, solver2_name, solver3_name):
        width = 2
        plt.semilogy(solver1.__error_cache[:, xi], linewidth=width, linestyle="solid")
        plt.semilogy(solver2.__error_cache[:, xi], linewidth=width, linestyle="solid")
        plt.semilogy(solver3.__error_cache[:, xi], linewidth=width, linestyle="solid")
        plt.title(f"comparison of {solver1_name}, {solver2_name} and {solver3_name} solvers xi_{xi} residual value")
        plt.ylabel(r"log(residual value)", fontsize=12)
        plt.xlabel(r"iteration", fontsize=12)
        plt.legend((f"{solver1_name}", f"{solver2_name}", f"-{solver3_name}"))
        plt.show()

    def plot_solution(self, solver):
        primal = self.__cache.get_primal()
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

