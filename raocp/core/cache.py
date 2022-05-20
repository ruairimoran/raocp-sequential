import numpy as np
import scipy.optimize
import raocp.core.problem_spec as ps
import raocp.core.cones as cones


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
        self.__old_primal = None
        self.__old_dual = None
        self.__initial_state = None

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
        self.__inverse_of_modified_control_dynamics = [np.zeros((0, 0))] * self.__num_nonleaf_nodes
        self.__sum_of_dynamics = [np.zeros((0, 0))] * self.__num_nodes  # A+BK

        # kernel projection
        self.__kernel_projection_operator = [np.zeros((0, 0))] * self.__num_nonleaf_nodes

        # populate arrays
        self._offline()

        # update cache with iteration zero
        self.update_cache()

    # GETTERS ##########################################################################################################

    def get_primal(self):
        return self.__primal, self.__old_primal

    def get_primal_split(self):
        return self.__split_p

    def get_dual(self):
        return self.__dual, self.__old_dual

    def get_dual_split(self):
        return self.__split_d

    # SETTERS ##########################################################################################################

    def cache_initial_state(self, state):
        self.__old_primal[0] = state
        self.__primal_cache[0][0] = state

    # CREATE ###########################################################################################################

    def _create_primal(self):
        self.__split_p = [None, 0,  # x = 1
                          self.__num_nodes,  # u = 2
                          self.__num_nodes + self.__num_nonleaf_nodes * 1,  # y = 3
                          self.__num_nodes + self.__num_nonleaf_nodes * 2,  # tau = 4
                          self.__num_nodes + self.__num_nonleaf_nodes * 2 + (self.__num_stages + 1),  # s = 5
                          self.__num_nodes + self.__num_nonleaf_nodes * 2 + (self.__num_stages + 1) * 2]  # end of 5
        self.__primal = [np.zeros(1)] * self.__split_p[-1]
        for i in range(self.__num_nodes):
            self.__primal[self.__split_p[1] + i] = np.zeros((self.__state_size, 1))
            if i < self.__num_nonleaf_nodes:
                self.__primal[self.__split_p[2] + i] = np.zeros((self.__control_size, 1))
                self.__primal[self.__split_p[3] + i] = np.zeros((2 * self.__raocp.tree.children_of(i).size + 1, 1))

        for i in range(self.__num_stages + 1):
            largest_node_at_stage = max(self.__raocp.tree.nodes_at_stage(i))
            # store variables in their node number inside the stage vector for s and tau
            self.__primal[self.__split_p[5] + i] = np.zeros((largest_node_at_stage + 1, 1))
            if i > 0:
                self.__primal[self.__split_p[4] + i] = np.zeros((largest_node_at_stage + 1, 1))

    def _create_dual(self):
        # parts 3, 4, 5, 6 of parent node put in children nodes for simple storage
        self.__split_d = [None, 0,  # start of part 1
                          self.__num_nonleaf_nodes * 1,  # start of part 2
                          self.__num_nonleaf_nodes * 2,  # start of part 3
                          self.__num_nonleaf_nodes * 2 + self.__num_nodes * 1,  # start of part 4
                          self.__num_nonleaf_nodes * 2 + self.__num_nodes * 2,  # start of part 5
                          self.__num_nonleaf_nodes * 2 + self.__num_nodes * 3,  # start of part 6
                          self.__num_nonleaf_nodes * 3 + self.__num_nodes * 3,  # start of part 7
                          self.__num_nonleaf_nodes * 3 + self.__num_nodes * 4,  # end of part 7
                          None, None,  # miss 8, 9, 10
                          self.__num_nonleaf_nodes * 3 + self.__num_nodes * 4,  # start of part 11
                          self.__num_nonleaf_nodes * 3 + self.__num_nodes * 5,  # start of part 12
                          self.__num_nonleaf_nodes * 3 + self.__num_nodes * 6,  # start of part 13
                          self.__num_nonleaf_nodes * 3 + self.__num_nodes * 7,  # start of part 14
                          self.__num_nonleaf_nodes * 3 + self.__num_nodes * 8]  # end of part 14
        self.__dual = [np.zeros(1)] * self.__split_d[-1]
        for i in range(1, self.__num_nodes):
            self.__dual[self.__split_d[3] + i] = np.zeros((self.__state_size, 1))
            self.__dual[self.__split_d[4] + i] = np.zeros((self.__control_size, 1))
            self.__dual[self.__split_d[7] + i] = np.zeros((self.__state_size, 1))
            if i >= self.__num_nonleaf_nodes:
                self.__dual[self.__split_d[11] + i] = np.zeros((self.__state_size, 1))
                self.__dual[self.__split_d[14] + i] = np.zeros((self.__state_size, 1))

    def _create_cones(self):
        self.__nonleaf_constraint_cone = [None] * self.__num_nonleaf_nodes
        self.__nonleaf_second_order_cone = [None] * self.__num_nodes
        self.__leaf_second_order_cone = [None] * self.__num_nodes
        for i in range(self.__num_nodes):
            if i < self.__num_nonleaf_nodes:
                self.__nonleaf_constraint_cone[i] = cones.Cartesian([cones.NonnegativeOrthant(),
                                                                     cones.NonnegativeOrthant()])
            if i > 0:
                self.__nonleaf_second_order_cone[i] = cones.SecondOrderCone()
            if i >= self.__num_nonleaf_nodes:
                self.__leaf_second_order_cone[i] = cones.SecondOrderCone()

    # CACHE ############################################################################################################

    def update_cache(self):
        """
        Update cache list of primal and dual and update 'old' parts to latest list
        """
        # primal
        self.__primal_cache.append(self.__primal.copy())
        self.__old_primal = self.__primal_cache[-1][:]

        # dual
        self.__dual_cache.append(self.__dual.copy())
        self.__old_dual = self.__dual_cache[-1][:]

    # OFFLINE ##########################################################################################################

    def _offline(self):
        """
        Upon creation of Cache class, calculate pre-computable arrays
        """
        self.offline_projection_dynamics()
        self.offline_projection_kernel()

    @staticmethod
    def inverse_using_cholesky(matrix):
        cholesky_of_matrix = np.linalg.cholesky(matrix)
        inverse_of_cholesky = np.linalg.inv(cholesky_of_matrix)
        inverse_of_matrix = inverse_of_cholesky.T @ inverse_of_cholesky
        return inverse_of_matrix

    def offline_projection_dynamics(self):
        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            self.__P[i] = np.eye(self.__state_size)

        for i in reversed(range(self.__num_nonleaf_nodes)):
            sum_for_modified_control_dynamics = 0
            sum_for_k = 0
            for j in self.__raocp.tree.children_of(i):
                sum_for_modified_control_dynamics += self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] \
                    @ self.__raocp.control_dynamics_at_node(j)
                sum_for_k += self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] \
                    @ self.__raocp.state_dynamics_at_node(j)

            self.__inverse_of_modified_control_dynamics[i] = \
                self.inverse_using_cholesky(np.eye(self.__control_size) + sum_for_modified_control_dynamics)
            self.__K[i] = - self.__inverse_of_modified_control_dynamics[i] @ sum_for_k
            sum_for_p = 0
            for j in self.__raocp.tree.children_of(i):
                self.__sum_of_dynamics[j] = self.__raocp.state_dynamics_at_node(j) \
                                    + self.__raocp.control_dynamics_at_node(j) @ self.__K[i]
                sum_for_p += self.__sum_of_dynamics[j].T @ self.__P[j] @ self.__sum_of_dynamics[j]

            self.__P[i] = np.eye(self.__state_size) + self.__K[i].T @ self.__K[i] + sum_for_p

    def offline_projection_kernel(self):
        for i in range(self.__num_nonleaf_nodes):
            eye = np.eye(len(self.__raocp.tree.children_of(i)))
            zeros = np.zeros((self.__raocp.risk_at_node(i).matrix_f.shape[1], eye.shape[0]))
            row1 = np.hstack((self.__raocp.risk_at_node(i).matrix_e.T, -eye, -eye))
            row2 = np.hstack((self.__raocp.risk_at_node(i).matrix_f.T, zeros, zeros))
            s2_space = np.vstack((row1, row2))
            kernel = scipy.linalg.null_space(s2_space)
            pseudoinverse_of_kernel = np.linalg.pinv(kernel)
            self.__kernel_projection_operator[i] = kernel @ pseudoinverse_of_kernel

    # ONLINE ###########################################################################################################

    # proximal of f ----------------------------------------------------------------------------------------------------

    def proximal_of_f(self, initial_state, solver_parameter):
        if self.__initial_state is None:
            self.__initial_state = initial_state
        self.proximal_of_relaxation_s_at_stage_zero(solver_parameter)
        self.project_on_dynamics()
        self.project_on_kernel()

    def proximal_of_relaxation_s_at_stage_zero(self, solver_parameter=1.0):
        """
        proximal operator of alpha * identity on s0
        """
        self.__primal[self.__split_p[5]] -= solver_parameter

    def project_on_dynamics(self):
        """
        use dynamic programming to project (x, u) onto the set S_1
        :returns: nothing
        """
        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            self.__q[i] = -2 * self.__q[i]

        for i in reversed(range(self.__num_nonleaf_nodes)):
            sum_for_d = 0
            for j in self.__raocp.tree.children_of(i):
                sum_for_d += self.__raocp.control_dynamics_at_node(j).T @ self.__q[i]

            self.__d[i] = self.__inverse_of_modified_control_dynamics[i] @ \
                (self.__primal[self.__split_p[2] + i] - sum_for_d)
            sum_for_q = 0
            for j in self.__raocp.tree.children_of(i):
                sum_for_q += self.__sum_of_dynamics[j].T @ \
                    (self.__P[j] @ self.__raocp.control_dynamics_at_node(j) @ self.__d[i] + self.__q[j])

            self.__q[i] = - self.__primal[self.__split_p[1] + i] + \
                self.__K[i].T @ (self.__d[i] - self.__primal[self.__split_p[2] + i]) + sum_for_q

        self.__primal[self.__split_p[1]] = self.__initial_state
        for i in range(self.__num_nonleaf_nodes):
            self.__primal[self.__split_p[2] + i] = self.__K[i] @ self.__primal[self.__split_p[1] + i] + self.__d[i]
            for j in self.__raocp.tree.children_of(i):
                self.__primal[self.__split_p[1] + j] = self.__sum_of_dynamics[j] @ \
                    self.__primal[self.__split_p[1] + i] + self.__raocp.control_dynamics_at_node(j) @ self.__d[i]

    def project_on_kernel(self):
        """
        use kernels to project (y, s, tau) onto the set S_2
        :returns: nothing
        """
        for i in range(self.__num_nonleaf_nodes):
            stage_at_children_of_i = self.__raocp.tree.stage_of(i) + 1
            children_of_i = self.__raocp.tree.children_of(i)
            # get children of i out of next stage of s and tau
            s_stack = self.__primal[self.__split_p[5] + stage_at_children_of_i][children_of_i[0]]
            tau_stack = self.__primal[self.__split_p[4] + stage_at_children_of_i][children_of_i[0]]
            if children_of_i.size > 1:
                for j in np.delete(children_of_i, 0):
                    s_stack = np.vstack((s_stack,
                                         self.__primal[self.__split_p[5] + stage_at_children_of_i][j]))
                    tau_stack = np.vstack((tau_stack,
                                           self.__primal[self.__split_p[4] + stage_at_children_of_i][j]))

            full_stack = np.vstack((self.__primal[self.__split_p[3] + i], s_stack, tau_stack))
            projection = self.__kernel_projection_operator[i] @ full_stack
            self.__primal[self.__split_p[3] + i] = projection[0: self.__primal[self.__split_p[3] + i].size]
            for k in range(children_of_i.size):
                self.__primal[self.__split_p[5] + stage_at_children_of_i][children_of_i[k]] = \
                    projection[self.__primal[self.__split_p[3] + i].size + k]
                self.__primal[self.__split_p[4] + stage_at_children_of_i][children_of_i[k]] = \
                    projection[self.__primal[self.__split_p[3] + i].size + children_of_i.size + k]

    # proximal of g conjugate ------------------------------------------------------------------------------------------

    def add_halves(self):
        self.__dual[self.__split_d[5]: self.__split_d[6]] = [j - 0.5 for j in self.__dual[self.__split_d[5]:
                                                                                          self.__split_d[6]]]
        self.__dual[self.__split_d[6]: self.__split_d[7]] = [j + 0.5 for j in self.__dual[self.__split_d[6]:
                                                                                          self.__split_d[7]]]
        self.__dual[self.__split_d[12]: self.__split_d[13]] = [j - 0.5 for j in self.__dual[self.__split_d[12]:
                                                                                            self.__split_d[13]]]
        self.__dual[self.__split_d[13]: self.__split_d[14]] = [j + 0.5 for j in self.__dual[self.__split_d[13]:
                                                                                            self.__split_d[14]]]

    def subtract_halves(self):
        self.__dual[self.__split_d[5]: self.__split_d[6]] = [j + 0.5 for j in self.__dual[self.__split_d[5]:
                                                                                          self.__split_d[6]]]
        self.__dual[self.__split_d[6]: self.__split_d[7]] = [j - 0.5 for j in self.__dual[self.__split_d[6]:
                                                                                          self.__split_d[7]]]
        self.__dual[self.__split_d[12]: self.__split_d[13]] = [j + 0.5 for j in self.__dual[self.__split_d[12]:
                                                                                            self.__split_d[13]]]
        self.__dual[self.__split_d[13]: self.__split_d[14]] = [j - 0.5 for j in self.__dual[self.__split_d[13]:
                                                                                            self.__split_d[14]]]

    def proximal_of_g_conjugate(self):  # not perfect ##################################################################
        # precomposition add halves
        self.add_halves()
        # proximal gbar (cone projections)
        for i in range(self.__num_nonleaf_nodes):
            [self.__dual[self.__split_d[1] + i], self.__dual[self.__split_d[2] + i]] = \
                self.__nonleaf_constraint_cone[i]\
                    .project([self.__dual[self.__split_d[1] + i], self.__dual[self.__split_d[2] + i]])
            children_of_i = self.__raocp.tree.children_of(i)
            for j in children_of_i:
                self.__dual[self.__split_d[3]: self.__split_d[7]][j] = self.__nonleaf_second_order_cone[j]\
                    .project(self.__dual[self.__split_d[3]: self.__split_d[7]][j])

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            self.__dual[self.__split_d[11]: self.__split_d[14]][i] = self.__leaf_second_order_cone[i]\
                .project(self.__dual[self.__split_d[11]: self.__split_d[14]][i])
        # precomposition subtract halves
        self.subtract_halves()
        # Moreau decomposition
        self.__dual = [a_i - b_i for a_i, b_i in zip(self.__old_dual, self.__dual)]
