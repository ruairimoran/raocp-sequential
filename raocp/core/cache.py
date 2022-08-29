import numpy as np
import scipy.optimize
from scipy.sparse.linalg import LinearOperator, eigs
import raocp.core.constraints.cones as cones
import raocp.core.raocp_spec as ps
import raocp.core.risks as risks
import raocp.core.operators as core_ops


class Cache:
    """
    Oracle of functions for solving RAOCPs using proximal algorithms
    """

    def __init__(self, problem_spec: ps.RAOCP):
        self.__raocp = problem_spec
        self.__num_nodes = self.__raocp.tree.num_nodes
        self.__num_nonleaf_nodes = self.__raocp.tree.num_nonleaf_nodes
        self.__num_leaf_nodes = self.__num_nodes - self.__num_nonleaf_nodes
        self.__num_stages = self.__raocp.tree.num_stages
        self.__state_size = self.__raocp.state_dynamics_at_node(1).shape[1]
        self.__control_size = self.__raocp.control_dynamics_at_node(1).shape[1]
        self.__primal_cache = []
        self.__dual_cache = []
        self.__template_primal = None
        self.__template_dual = None
        self.__initial_state = None
        self.__parameter_1 = None
        self.__parameter_2 = None
        self.__linear_operator = None
        self.__error = [np.zeros(1)] * 3
        self.__error_cache = None
        self.__old_ell_prim = None
        self.__old_ell_t_dual = None

        # create primal list
        self._create_primal()

        # create dual list / parts 3,4,5,6 stored in child nodes for convenience
        self._create_dual()

        # create cones
        self._create_cones()

        # dynamics projection
        self.__P = [np.zeros((self.__state_size, self.__state_size))] * self.__num_nodes
        self.__q = [np.zeros((self.__state_size, 1))] * self.__num_nodes
        self.__K = [np.zeros((self.__state_size, self.__state_size))] * self.__num_nonleaf_nodes
        self.__d = [np.zeros((self.__state_size, 1))] * self.__num_nonleaf_nodes
        self.__cholesky_data = [(None, None)] * self.__num_nonleaf_nodes
        self.__sum_of_dynamics = [np.zeros((0, 0))] * self.__num_nodes  # A+BK

        # kernel projection
        self.__kernel_constraint_matrix = [np.zeros((0, 0))] * self.__num_nonleaf_nodes
        self.__null_space_matrix = [np.zeros((0, 0))] * self.__num_nonleaf_nodes

        # populate arrays
        self._offline()

        # update cache and templates with iteration zero
        self.update_cache()
        self.update_templates()

    # GETTERS ##########################################################################################################

    def get_raocp(self):
        return self.__raocp

    def get_primal(self):
        return self.__primal.copy()

    def get_primal_segments(self):
        return self.__segment_p.copy()

    def get_dual(self):
        return self.__dual.copy()

    def get_dual_segments(self):
        return self.__segment_d.copy()

    def get_primal_and_dual(self):
        return self.__primal.copy(), self.__dual.copy()

    def get_kernel_constraint_matrices(self):
        return self.__kernel_constraint_matrix.copy()

    def get_nullspace_matrices(self):
        return self.__null_space_matrix.copy()

    def get_error_cache(self):
        return self.__error_cache

    # SETTERS ##########################################################################################################

    def cache_initial_state(self, state):
        self.__initial_state = state
        self.__primal_cache[0][0] = state

    def set_primal(self, candidate_primal):
        primal_length = len(self.__template_primal)
        if len(candidate_primal) != primal_length:
            raise Exception("Candidate primal list is wrong length")
        for i in range(primal_length):
            if candidate_primal[i].shape != self.__template_primal[i].shape:
                all_segments = range(1, 6)
                for s in reversed(all_segments):
                    if i >= self.__segment_d[s]:
                        segment = s
                        node = i - self.__segment_d[s]
                        break

                raise Exception(f"Candidate primal array shape error in segment {segment} at node {node},\n"
                                f"candidate shape: {candidate_primal[i].shape},\n"
                                f"current shape: {self.__template_primal[i].shape}")
            else:
                self.__primal[i] = candidate_primal[i].copy()

    def set_dual(self, candidate_dual):
        dual_length = len(self.__dual_cache[0])
        if len(candidate_dual) != dual_length:
            raise Exception("Candidate dual list is wrong length")
        for i in range(dual_length):
            if candidate_dual[i].shape != self.__template_dual[i].shape:
                all_segments = range(1, 15)
                exclude = {8, 9, 10}
                active_segments = [num for num in all_segments if num not in exclude]
                for s in reversed(active_segments):
                    if i >= self.__segment_d[s]:
                        segment = s
                        node = i - self.__segment_d[s]
                        break

                raise Exception(f"Candidate dual array shape error in segment {segment} at node {node},\n"
                                f"candidate shape: {candidate_dual[i].shape},\n"
                                f"current shape: {self.__template_dual[i].shape}")
            else:
                self.__dual[i] = candidate_dual[i].copy()

    def set_linear_operator(self, linear_operator):
        self.__linear_operator = linear_operator

    # CREATE ###########################################################################################################

    def _create_primal(self):
        self.__segment_p = [None, 0,  # x = 1
                            self.__num_nodes * 1,  # u = 2
                            self.__num_nodes * 1 + self.__num_nonleaf_nodes * 1,  # y = 3
                            self.__num_nodes * 1 + self.__num_nonleaf_nodes * 2,  # tau = 4
                            self.__num_nodes * 2 + self.__num_nonleaf_nodes * 2,  # s = 5
                            self.__num_nodes * 3 + self.__num_nonleaf_nodes * 2]  # end of 5
        self.__primal = [np.zeros((1, 1))] * self.__segment_p[-1]
        for i in range(self.__num_nodes):
            self.__primal[self.__segment_p[1] + i] = np.zeros((self.__state_size, 1))
            if i < self.__num_nonleaf_nodes:
                self.__primal[self.__segment_p[2] + i] = np.zeros((self.__control_size, 1))
                self.__primal[self.__segment_p[3] + i] = np.zeros((2 * self.__raocp.tree.children_of(i).size + 1, 1))

    def _create_dual(self):
        # parts 3, 4, 5 and 6 of parent node put in children nodes for simple storage
        self.__segment_d = [None, 0,  # start of part 1
                            self.__num_nodes * 1,  # start of part 2
                            self.__num_nodes * 2,  # start of part 3
                            self.__num_nodes * 3,  # start of part 4
                            self.__num_nodes * 4,  # start of part 5
                            self.__num_nodes * 5,  # start of part 6
                            self.__num_nodes * 6,  # start of part 7
                            self.__num_nodes * 7,  # end of part 7
                            None, None,  # miss 8, 9, 10
                            self.__num_nodes * 7,  # start of part 11
                            self.__num_nodes * 8,  # start of part 12
                            self.__num_nodes * 9,  # start of part 13
                            self.__num_nodes * 10,  # start of part 14
                            self.__num_nodes * 11]  # end of part 14
        self.__dual = [np.zeros((1, 1))] * self.__segment_d[-1]
        for i in range(self.__num_nodes):
            if i < self.__num_nonleaf_nodes:
                self.__dual[self.__segment_d[1] + i] = np.zeros((2 * self.__raocp.tree.children_of(i).size + 1, 1))
                if self.__raocp.nonleaf_constraint_at_node(i).is_active:
                    self.__dual[self.__segment_d[7] + i] = np.zeros((self.__raocp.nonleaf_constraint_at_node(i)
                                                                     .state_matrix.shape[0], 1))
            if i > 0:
                self.__dual[self.__segment_d[3] + i] = np.zeros((self.__state_size, 1))
                self.__dual[self.__segment_d[4] + i] = np.zeros((self.__control_size, 1))
            if i >= self.__num_nonleaf_nodes:
                self.__dual[self.__segment_d[11] + i] = np.zeros((self.__state_size, 1))
                if self.__raocp.leaf_constraint_at_node(i).is_active:
                    self.__dual[self.__segment_d[14] + i] = np.zeros((self.__raocp.leaf_constraint_at_node(i)
                                                                      .state_matrix.shape[0], 1))

    def _create_cones(self):
        self.__nonleaf_dual_cone = []
        for i in range(self.__num_nonleaf_nodes):
            if type(self.__raocp.risk_at_node(i)) is risks.AVaR:
                self.__nonleaf_dual_cone.append(self.__raocp.risk_at_node(i).cone)
            else:
                raise Exception(f"Risk at node {i} not defined")

        self.__nonleaf_nonnegorth_cone = [cones.NonnegativeOrthant()] * self.__num_nonleaf_nodes
        self.__nonleaf_second_order_cone = [cones.SecondOrderCone()] * self.__num_nodes
        self.__leaf_second_order_cone = [cones.SecondOrderCone()] * self.__num_nodes

    # CACHE ############################################################################################################

    def update_cache(self):
        """
        Update cache list of primal and dual
        """
        # primal
        self.__primal_cache.append(self.__primal.copy())

        # dual
        self.__dual_cache.append(self.__dual.copy())

    # TEMPLATES ########################################################################################################

    def update_templates(self):
        """
        Update templates of primal and dual
        """
        # primal
        self.__template_primal = self.__primal.copy()

        # dual
        self.__template_dual = self.__dual.copy()

    # OFFLINE ##########################################################################################################

    def _offline(self):
        """
        Upon creation of Cache class, calculate pre-computable arrays
        """
        self.offline_projection_dynamics()
        self.offline_projection_kernel()

    def offline_projection_dynamics(self):
        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            self.__P[i] = np.eye(self.__state_size)

        for i in reversed(range(self.__num_nonleaf_nodes)):
            children_of_i = self.__raocp.tree.children_of(i)
            sum_for_r = 0
            sum_for_k = 0
            for j in children_of_i:
                sum_for_r = sum_for_r + \
                    self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] @ \
                    self.__raocp.control_dynamics_at_node(j)
                sum_for_k = sum_for_k + \
                    self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] @ \
                    self.__raocp.state_dynamics_at_node(j)

            r_tilde = np.eye(self.__control_size) + sum_for_r
            self.__cholesky_data[i] = scipy.linalg.cho_factor(r_tilde)
            self.__K[i] = scipy.linalg.cho_solve(self.__cholesky_data[i], -sum_for_k)

            sum_for_p = 0
            for j in children_of_i:
                self.__sum_of_dynamics[j] = self.__raocp.state_dynamics_at_node(j) \
                                    + self.__raocp.control_dynamics_at_node(j) @ self.__K[i]
                sum_for_p = sum_for_p + self.__sum_of_dynamics[j].T @ self.__P[j] @ self.__sum_of_dynamics[j]

            self.__P[i] = np.eye(self.__state_size) + self.__K[i].T @ self.__K[i] + sum_for_p

    def offline_projection_kernel(self):
        for i in range(self.__num_nonleaf_nodes):
            eye = np.eye(len(self.__raocp.tree.children_of(i)))
            zeros = np.zeros((self.__raocp.risk_at_node(i).matrix_f.shape[1], eye.shape[0]))
            row1 = np.hstack((self.__raocp.risk_at_node(i).matrix_e.T, -eye, -eye))
            row2 = np.hstack((self.__raocp.risk_at_node(i).matrix_f.T, zeros, zeros))
            self.__kernel_constraint_matrix[i] = np.vstack((row1, row2))
            self.__null_space_matrix[i] = scipy.linalg.null_space(self.__kernel_constraint_matrix[i])

    # ONLINE ###########################################################################################################

    # proximal of f ----------------------------------------------------------------------------------------------------

    def proximal_of_primal(self, solver_parameter):
        self.proximal_of_relaxation_s_at_stage_zero(solver_parameter)
        self.project_on_dynamics()
        self.project_on_kernel()

    def proximal_of_relaxation_s_at_stage_zero(self, solver_parameter):
        """
        proximal operator of alpha * identity on s0
        """
        self.__primal[self.__segment_p[5]] -= solver_parameter

    def project_on_dynamics(self):
        """
        use dynamic programming to project (x, u) onto the dynamics set S_1
        :returns: nothing
        """
        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            self.__q[i] = -self.__primal[self.__segment_p[1] + i]

        for i in reversed(range(self.__num_nonleaf_nodes)):
            sum_for_d = 0
            for j in self.__raocp.tree.children_of(i):
                sum_for_d += self.__raocp.control_dynamics_at_node(j).T @ self.__q[j]

            self.__d[i] = scipy.linalg.cho_solve(self.__cholesky_data[i],
                                                 self.__primal[self.__segment_p[2] + i] - sum_for_d)
            sum_for_q = 0
            for j in self.__raocp.tree.children_of(i):
                sum_for_q += self.__sum_of_dynamics[j].T @ \
                    (self.__P[j] @ self.__raocp.control_dynamics_at_node(j) @ self.__d[i] + self.__q[j])

            self.__q[i] = -self.__primal[self.__segment_p[1] + i] + \
                self.__K[i].T @ (self.__d[i] - self.__primal[self.__segment_p[2] + i]) + sum_for_q

        self.__primal[self.__segment_p[1]] = self.__initial_state
        for i in range(self.__num_nonleaf_nodes):
            self.__primal[self.__segment_p[2] + i] = self.__K[i] @ self.__primal[self.__segment_p[1] + i] + self.__d[i]
            for j in self.__raocp.tree.children_of(i):
                self.__primal[self.__segment_p[1] + j] = self.__sum_of_dynamics[j] @ \
                                                         self.__primal[self.__segment_p[1] + i] + \
                                                         self.__raocp.control_dynamics_at_node(j) @ self.__d[i]

    def project_on_kernel(self):
        """
        use kernels to project (y, tau+, s+) onto the kernel set S_2
        :returns: nothing
        """
        for i in range(self.__num_nonleaf_nodes):
            children_of_i = self.__raocp.tree.children_of(i)
            y = self.__primal[self.__segment_p[3] + i]
            y_size = y.size
            t_or_s_size = children_of_i.size  # size of tau which is same as size of s
            # get children of i out of tau and s
            t_stack = self.__primal[self.__segment_p[4] + children_of_i[0]]
            s_stack = self.__primal[self.__segment_p[5] + children_of_i[0]]
            if t_or_s_size > 1:
                for j in np.delete(children_of_i, 0):
                    t_stack = np.vstack((t_stack, self.__primal[self.__segment_p[4] + j]))
                    s_stack = np.vstack((s_stack, self.__primal[self.__segment_p[5] + j]))

            full_stack = np.vstack((y, t_stack, s_stack))
            projection = self.__null_space_matrix[i] @ \
                np.linalg.lstsq(self.__null_space_matrix[i], full_stack, rcond=None)[0]
            if not np.allclose(np.linalg.norm(self.__kernel_constraint_matrix[i] @ projection, np.inf), 0):
                raise Exception("Kernel projection error")
            self.__primal[self.__segment_p[3] + i] = projection[0: y_size]
            for j in range(t_or_s_size):
                self.__primal[self.__segment_p[4] + children_of_i[j]] = (projection[y_size + j]).reshape(-1, 1)
                self.__primal[self.__segment_p[5] + children_of_i[j]] = (projection[y_size + t_or_s_size + j]) \
                    .reshape(-1, 1)

    # proximal of g conjugate ------------------------------------------------------------------------------------------

    def proximal_of_dual(self, solver_parameter):
        self.modify_dual(solver_parameter)
        self.add_halves()
        modified_dual = self.__dual.copy()  # take copy of modified dual
        self.project_on_constraints_nonleaf()
        self.project_on_constraints_leaf()
        self.modify_projection(solver_parameter, modified_dual)

    def modify_dual(self, solver_parameter):
        # algo 6
        for i in range(len(self.__dual)):
            self.__dual[i] = self.__dual[i] / solver_parameter

    def add_halves(self):
        plus_half = 0.5
        minus_half = -0.5
        for i in range(self.__segment_d[5], self.__segment_d[6]):
            self.__dual[i] = self.__dual[i] + minus_half

        for i in range(self.__segment_d[6], self.__segment_d[7]):
            self.__dual[i] = self.__dual[i] + plus_half

        for i in range(self.__segment_d[12], self.__segment_d[13]):
            self.__dual[i] = self.__dual[i] + minus_half

        for i in range(self.__segment_d[13], self.__segment_d[14]):
            self.__dual[i] = self.__dual[i] + plus_half

    def project_on_constraints_nonleaf(self):
        # algo 7
        for i in range(self.__num_nonleaf_nodes):
            self.__dual[self.__segment_d[1] + i] = self.__nonleaf_dual_cone[i] \
                .project_onto_dual([self.__dual[self.__segment_d[1] + i]])
            self.__dual[self.__segment_d[2] + i] = self.__nonleaf_nonnegorth_cone[i] \
                .project(self.__dual[self.__segment_d[2] + i])
            for j in self.__raocp.tree.children_of(i):
                start = 3
                end = 6 + 1  # +1 for loops
                size = [0] * end
                for k in range(start, end):
                    size[k] = size[k-1] + self.__dual[self.__segment_d[k] + j].size

                soc_vector = np.vstack((self.__dual[self.__segment_d[3] + j], self.__dual[self.__segment_d[4] + j],
                                        self.__dual[self.__segment_d[5] + j], self.__dual[self.__segment_d[6] + j]))
                soc_projection = self.__nonleaf_second_order_cone[j].project(soc_vector)
                for k in range(start, end):
                    self.__dual[self.__segment_d[k] + j] = soc_projection[size[k - 1]: size[k]]

            if self.__raocp.nonleaf_constraint_at_node(i).is_active:
                self.__dual[self.__segment_d[7] + i] = self.__raocp.nonleaf_constraint_at_node(i) \
                    .project(self.__dual[self.__segment_d[7] + i])

    def project_on_constraints_leaf(self):
        # algo 7
        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            start = 11
            end = 13 + 1  # +1 for loops
            size = [0] * end
            for j in range(start, end):
                size[j] = size[j - 1] + self.__dual[self.__segment_d[j] + i].size

            soc_vector = np.vstack((self.__dual[self.__segment_d[11] + i], self.__dual[self.__segment_d[12] + i],
                                    self.__dual[self.__segment_d[13] + i]))
            soc_projection = self.__leaf_second_order_cone[i].project(soc_vector)
            for j in range(start, end):
                self.__dual[self.__segment_d[j] + i] = soc_projection[size[j - 1]: size[j]]

            if self.__raocp.leaf_constraint_at_node(i).is_active:
                self.__dual[self.__segment_d[14] + i] = self.__raocp.leaf_constraint_at_node(i) \
                    .project(self.__dual[self.__segment_d[14] + i])

    def modify_projection(self, solver_parameter, modified_dual):
        self.__dual = [solver_parameter * (a_i - b_i) for a_i, b_i in zip(modified_dual, self.__dual)]

    # CHAMBOLLE-POCK OPERATOR ##########################################################################################

    def chock_operator(self, current_prim, current_dual):
        # run primal part of algorithm -----------------------------------------------------------
        # get memory space for ell_transpose_dual
        ell_transpose_dual = self.get_primal()
        # operate L transpose on dual and store in ell_transpose_dual
        self.__linear_operator.ell_transpose(current_dual, ell_transpose_dual)
        # old primal minus (alpha1 times ell_transpose_dual)
        new_primal = [a_i - b_i for a_i, b_i in zip(current_prim, [j * self.__parameter_1
                                                                   for j in ell_transpose_dual])]
        self.set_primal(new_primal)
        self.proximal_of_primal(self.__parameter_1)
        # run dual part of algorithm -------------------------------------------------------------
        mod_prim_ = self.get_primal()
        # get memory space for ell_transpose_dual
        ell_primal = self.get_dual()
        # two times new primal minus old primal
        modified_primal = [a_i - b_i for a_i, b_i in zip([j * 2 for j in mod_prim_], current_prim)]
        # operate L on modified primal
        self.__linear_operator.ell(modified_primal, ell_primal)
        # old dual plus (gamma times ell_primal)
        new_dual = [a_i + b_i for a_i, b_i in zip(current_dual, [j * self.__parameter_2
                                                                 for j in ell_primal])]
        self.set_dual(new_dual)
        self.proximal_of_dual(self.__parameter_2)
        # get new parts
        new_prim_, new_dual_ = self.get_primal_and_dual()
        return new_prim_, new_dual_, ell_primal, ell_transpose_dual

    # HELPER FUNCTIONS #################################################################################################

    def get_parameters(self):
        # find alpha_1 and _2
        prim, dual = self.get_primal_and_dual()
        size_prim = np.vstack(prim).size
        size_dual = np.vstack(dual).size
        ell = LinearOperator(dtype=None, shape=(size_dual, size_prim),
                             matvec=self.__linear_operator.linop_ell)
        ell_transpose = LinearOperator(dtype=None, shape=(size_prim, size_dual),
                                       matvec=self.__linear_operator.linop_ell_transpose)
        ell_transpose_ell = ell_transpose * ell
        eigens, _ = eigs(ell_transpose_ell)
        ell_norm = np.real(max(eigens))
        one_over_norm = 0.999 / ell_norm
        self.__parameter_1 = one_over_norm
        self.__parameter_2 = one_over_norm

    @staticmethod
    def parts_to_vector(prim_, dual_):
        return np.vstack((np.vstack(prim_), np.vstack(dual_)))

    def vector_to_parts(self, vector_):
        prim_, dual_ = self.get_primal_and_dual()
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

    def get_chock_norm(self, vector):
        norm = np.sqrt(self.get_chock_inner_prod(vector, vector))
        return norm

    def get_chock_norm_squared(self, vector):
        norm = self.get_chock_inner_prod(vector, vector)
        return norm

    def get_chock_inner_prod(self, vector_a, vector_b):
        if vector_a.shape[1] != 1 or vector_b.shape[1] != 1:
            raise Exception("non column vectors provided to inner product")
        inner = vector_a.T @ self.get_chock_inner_prod_matrix(vector_b)
        return inner[0]

    def get_chock_inner_prod_matrix(self, vector):
        prim, dual = self.vector_to_parts(vector)
        ell_transpose_dual, ell_prim = self.get_primal_and_dual()
        self.__linear_operator.ell_transpose(dual, ell_transpose_dual)
        self.__linear_operator.ell(prim, ell_prim)
        modified_prim = [a_i - (self.__parameter_1 * b_i) for a_i, b_i in zip(prim, ell_transpose_dual)]
        modified_dual = [a_i - (self.__parameter_2 * b_i) for a_i, b_i in zip(dual, ell_prim)]
        modified_vector = self.parts_to_vector(modified_prim, modified_dual)
        return modified_vector

    # ERROR HANDLING ###################################################################################################
    def get_current_error(self, new_prim_, old_prim_, new_dual_, old_dual_, ell_prim_, ell_t_dual_):
        # calculate error
        xi_0, xi_1, xi_2 = self.calculate_chock_errors(new_prim_, old_prim_, new_dual_, old_dual_,
                                                       ell_prim_, ell_t_dual_)
        if xi_0 is not None:
            xi = [xi_0, xi_1, xi_2]
            for i in range(3):
                inf_norm_xi = [np.linalg.norm(a_i, ord=np.inf) for a_i in xi[i]]
                self.__error[i] = np.linalg.norm(inf_norm_xi, np.inf)

            return max(self.__error)
        else:
            return np.inf

    def calculate_chock_errors(self, new_prim_, old_prim_, new_dual_, old_dual_, ell_prim_, ell_t_dual_):
        # in this function, p = primal and d = dual
        if self.__old_ell_prim is not None and self.__old_ell_t_dual is not None:
            # error 1
            p_minus_p_new_over_alpha1 = [(a_i - b_i) / self.__parameter_1 for a_i, b_i in zip(old_prim_, new_prim_)]
            ell_transpose_d_minus_d_new = [a_i - b_i for a_i, b_i in zip(self.__old_ell_t_dual, ell_t_dual_)]
            xi_1 = [a_i - b_i for a_i, b_i in zip(p_minus_p_new_over_alpha1, ell_transpose_d_minus_d_new)]
            # error 2
            d_minus_d_new_over_alpha2 = [(a_i - b_i) / self.__parameter_2 for a_i, b_i in zip(old_dual_, new_dual_)]
            ell_p_new_minus_p = [a_i - b_i for a_i, b_i in zip(ell_prim_, self.__old_ell_prim)]
            xi_2 = [a_i + b_i for a_i, b_i in zip(d_minus_d_new_over_alpha2, ell_p_new_minus_p)]
            # error 0
            ell_transpose_error2 = self.get_primal()  # get memory position
            self.__linear_operator.ell_transpose(xi_2, ell_transpose_error2)
            xi_0 = [a_i + b_i for a_i, b_i in zip(xi_1, ell_transpose_error2)]
            # reset
            self.__old_ell_prim = ell_prim_
            self.__old_ell_t_dual = ell_t_dual_
        else:
            self.__old_ell_prim = ell_prim_
            self.__old_ell_t_dual = ell_t_dual_
            xi_0 = None
            xi_1 = None
            xi_2 = None
        return xi_0, xi_1, xi_2

    def cache_errors(self, i_):
        if i_ == 0:
            self.__error_cache = self.__error
        else:
            self.__error_cache = np.vstack((self.__error_cache, np.array(self.__error)))
