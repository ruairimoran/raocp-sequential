import unittest
import numpy as np
import cvxpy as cp
import raocp.core.cache as core_cache
import raocp.core.costs as core_costs
import raocp.core.dynamics as core_dynamics
import raocp.core.nodes as core_nodes
import raocp.core.raocp_spec as core_spec
import raocp.core.risks as core_risks
import raocp.core.scenario_tree as core_tree
import raocp.core.constraints.rectangle as rectangle
from copy import deepcopy


class TestCache(unittest.TestCase):
    __tree_from_markov = None
    __raocp_from_markov = None
    __good_size = 3

    @staticmethod
    def _construct_tree_from_markov():
        if TestCache.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.4, 0.1])
            (N, tau) = (4, 3)
            TestCache.__tree_from_markov = core_tree.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

    @staticmethod
    def _construct_raocp_from_markov():
        if TestCache.__raocp_from_markov is None:
            tree = TestCache.__tree_from_markov

            # construct markovian set of system and control dynamics
            system = np.eye(TestCache.__good_size)
            set_system = [system, 2 * system, 3 * system]  # n x n matrices
            control = np.eye(TestCache.__good_size)
            set_control = [control, 2 * control, 3 * control]  # n x u matrices
            dynamics = [core_dynamics.Dynamics(set_system[0], set_control[0]),
                        core_dynamics.Dynamics(set_system[1], set_control[1]),
                        core_dynamics.Dynamics(set_system[2], set_control[2])]

            # construct cost weight matrices
            nl = core_nodes.Nonleaf()
            l = core_nodes.Leaf()
            nonleaf_state_weight = 10 * np.eye(2)  # n x n matrix
            nonleaf_state_weights = [nonleaf_state_weight, 2 * nonleaf_state_weight, 3 * nonleaf_state_weight]
            control_weight = np.eye(2)  # u x u matrix OR scalar
            control_weights = [control_weight, 2 * control_weight, 3 * control_weight]
            nonleaf_costs = [core_costs.Quadratic(nl, nonleaf_state_weights[0], control_weights[0]),
                             core_costs.Quadratic(nl, nonleaf_state_weights[1], control_weights[1]),
                             core_costs.Quadratic(nl, nonleaf_state_weights[2], control_weights[2]), ]
            leaf_state_weight = 5 * np.eye(2)  # n x n matrix
            leaf_costs = core_costs.Quadratic(l, leaf_state_weight)

            # construct constraint min and max
            nonleaf_size = TestCache.__good_size + TestCache.__good_size
            leaf_size = TestCache.__good_size
            nl_min = -2 * np.ones((nonleaf_size, 1))
            nl_max = 2 * np.ones((nonleaf_size, 1))
            l_min = -0.5 * np.ones((leaf_size, 1))
            l_max = 0.5 * np.ones((leaf_size, 1))
            nl_rect = rectangle.Rectangle(nl, nl_min, nl_max)
            l_rect = rectangle.Rectangle(l, l_min, l_max)

            # define risks
            alpha = 0.5
            risks = core_risks.AVaR(alpha)

            TestCache.__raocp_from_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_markovian_dynamics(dynamics) \
                .with_markovian_nonleaf_costs(nonleaf_costs) \
                .with_all_leaf_costs(leaf_costs) \
                .with_all_nonleaf_constraints(nl_rect) \
                .with_all_leaf_constraints(l_rect) \
                .with_all_risks(risks)

    @staticmethod
    def _construct_mock_cache():
        mock_cache = core_cache.Cache(TestCache.__raocp_from_markov)
        primal_segments = mock_cache.get_primal_segments()
        dual_segments = mock_cache.get_dual_segments()
        return mock_cache, primal_segments, dual_segments

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestCache._construct_tree_from_markov()
        TestCache._construct_raocp_from_markov()

    def test_set_primal(self):
        mock_cache, seg_p, seg_d = self._construct_mock_cache()
        prim = mock_cache.get_primal()  # template
        length = len(prim)
        primal_set = deepcopy(prim)
        for i in range(length):
            primal_set[i] = np.random.randn(prim[i].size).reshape(-1, 1)

        mock_cache.set_primal(primal_set)
        primal_get = mock_cache.get_primal()
        for i in range(length):
            self.assertTrue(np.array_equal(primal_set[i], primal_get[i]))

    def test_set_dual(self):
        mock_cache, seg_p, seg_d = self._construct_mock_cache()
        dual = mock_cache.get_dual()  # template
        length = len(dual)
        dual_set = deepcopy(dual)
        for i in range(length):
            dual_set[i] = np.random.randn(dual[i].size).reshape(-1, 1)

        mock_cache.set_dual(dual_set)
        dual_get = mock_cache.get_dual()
        for i in range(length):
            self.assertTrue(np.array_equal(dual_set[i], dual_get[i]))

    def test_cache_initial_state(self):
        mock_cache, seg_p, seg_d = self._construct_mock_cache()
        prim = mock_cache.get_primal()  # template
        prim[seg_p[1]] = np.random.randn(prim[seg_p[1]].size).reshape(-1, 1)
        mock_cache.cache_initial_state(prim[seg_p[1]])
        self.assertTrue(np.array_equal(prim[seg_p[1]], mock_cache._Cache__initial_state))
        self.assertTrue(np.array_equal(prim[seg_p[1]], mock_cache._Cache__primal_cache[0][0]))

    def test_proximal_of_relaxation_s_at_stage_zero(self):
        mock_cache, seg_p, seg_d = self._construct_mock_cache()
        # s = segment 5
        s = 5
        prim = mock_cache.get_primal()  # get old primal as template
        parameter = np.random.randn(prim[seg_p[s]].size)
        mock_cache.proximal_of_relaxation_s_at_stage_zero(parameter)
        prim = mock_cache.get_primal()  # get modified primal
        self.assertEqual(-parameter, prim[seg_p[s]][0])

    def test_project_on_dynamics(self):
        mock_cache, seg_p, _ = self._construct_mock_cache()
        prim = mock_cache.get_primal()  # template
        for i in range(seg_p[1], seg_p[3]):
            prim[i] = np.random.randn(prim[i].size).reshape(-1, 1)

        # solve with dp
        mock_cache.cache_initial_state(prim[seg_p[1]])
        mock_cache.set_primal(prim)
        mock_cache.project_on_dynamics()
        dp_result = mock_cache.get_primal()
        x_dp = np.asarray(dp_result[seg_p[1]: seg_p[2]])[:, :, 0]
        u_dp = np.asarray(dp_result[seg_p[2]: seg_p[3]])[:, :, 0]
        # ensure x0 stayed the same
        self.assertTrue(np.allclose(prim[seg_p[1]].T, x_dp[0]))

        # solve with cvxpy
        for i in range(seg_p[1], seg_p[3]):
            prim[i] = prim[i].reshape(-1,)

        x_bar = np.asarray(prim[seg_p[1]: seg_p[2]])
        u_bar = np.asarray(prim[seg_p[2]: seg_p[3]])
        N = self.__tree_from_markov.num_nodes
        n = self.__tree_from_markov.num_nonleaf_nodes
        x = cp.Variable(x_bar.shape)
        u = cp.Variable(u_bar.shape)
        # sum problem objectives and concatenate constraints
        cost = 0
        constraints = [x[0] == x_bar[0]]
        # nonleaf nodes
        for node in range(n):
            cost += cp.sum_squares(x[node] - x_bar[node]) + cp.sum_squares(u[node] - u_bar[node])
            for ch in self.__tree_from_markov.children_of(node):
                constraints += [x[ch] ==
                                self.__raocp_from_markov.state_dynamics_at_node(ch) @ x[node] +
                                self.__raocp_from_markov.control_dynamics_at_node(ch) @ u[node]]

        # leaf nodes
        for node in range(n, N):
            cost += cp.sum_squares(x[node] - x_bar[node])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        # ensure x0 stayed the same
        self.assertTrue(np.allclose(prim[seg_p[1]], x.value[0]))

        # check solutions are similar
        self.assertTrue(np.allclose(x.value, x_dp))
        self.assertTrue(np.allclose(u.value, u_dp))

    def test_kernel_projection(self):
        mock_cache, seg_p, _ = self._construct_mock_cache()
        prim = mock_cache.get_primal()  # template
        for i in range(seg_p[3], seg_p[6]):
            if i == seg_p[4] or i == seg_p[5]:
                pass
            else:
                prim[i] = np.random.randn(prim[i].size).reshape(-1, 1)

        mock_cache.set_primal(prim)
        mock_cache.project_on_kernel()
        proj = mock_cache.get_primal()
        constraint_matrix = mock_cache.get_kernel_constraint_matrices()
        nullspace_matrix = mock_cache.get_nullspace_matrices()
        for i in range(self.__tree_from_markov.num_nonleaf_nodes):
            children = self.__tree_from_markov.children_of(i)

            t_stack = proj[seg_p[4] + children[0]]
            s_stack = proj[seg_p[5] + children[0]]
            if children.size > 1:
                for j in np.delete(children, 0):
                    t_stack = np.vstack((t_stack, proj[seg_p[4] + j]))
                    s_stack = np.vstack((s_stack, proj[seg_p[5] + j]))

            proj_stack = np.vstack((proj[seg_p[3] + i], t_stack, s_stack))
            kernel = constraint_matrix[i] @ proj_stack
            inf_norm = np.linalg.norm(kernel, np.inf)
            self.assertTrue(np.allclose(inf_norm, 0))

            # run in cvxpy
            cp_t_stack = prim[seg_p[4] + children[0]]
            cp_s_stack = prim[seg_p[5] + children[0]]
            if children.size > 1:
                for j in np.delete(children, 0):
                    cp_t_stack = np.vstack((cp_t_stack, prim[seg_p[4] + j]))
                    cp_s_stack = np.vstack((cp_s_stack, prim[seg_p[5] + j]))

            cp_stack = np.vstack((prim[seg_p[3] + i], cp_t_stack, cp_s_stack)).reshape(-1,)
            minimiser = cp.Variable(nullspace_matrix[i].shape[1])
            cost = cp.sum_squares(nullspace_matrix[i] @ minimiser - cp_stack)
            prob = cp.Problem(cp.Minimize(cost))
            prob.solve()
            cp_proj = nullspace_matrix[i] @ minimiser.value
            cp_kernel = constraint_matrix[i] @ cp_proj
            cp_inf_norm = np.linalg.norm(cp_kernel, np.inf)
            self.assertTrue(np.allclose(cp_inf_norm, 0))

            # check against each other
            self.assertTrue(np.allclose(proj_stack.reshape(-1,), cp_proj))

    def test_modify_dual(self):
        mock_cache, _, seg_d = self._construct_mock_cache()
        parameter = 0
        while parameter == 0:
            parameter = np.linalg.norm(np.random.randn(1), 2)

        dual = mock_cache.get_dual()
        for i in range(seg_d[1], seg_d[15]):
            dual[i] = np.random.randn(dual[i].size).reshape(-1, 1)

        mock_cache.set_dual(dual)
        mock_cache.modify_dual(parameter)
        new_dual = mock_cache.get_dual()
        for i in range(seg_d[1], seg_d[15]):
            test_result = dual[i] / parameter
            self.assertTrue(np.allclose(test_result, new_dual[i]))

    def test_add_halves(self):
        mock_cache, _, seg_d = self._construct_mock_cache()
        mock_cache.add_halves()
        dual = mock_cache.get_dual()
        minus_half = -0.5
        plus_half = 0.5
        for i in dual[seg_d[5]: seg_d[6]]:
            self.assertEqual(i, minus_half)

        for i in dual[seg_d[6]: seg_d[7]]:
            self.assertEqual(i, plus_half)

        for i in dual[seg_d[12]: seg_d[13]]:
            self.assertEqual(i, minus_half)

        for i in dual[seg_d[13]: seg_d[14]]:
            self.assertEqual(i, plus_half)

    def test_project_on_constraints_nonleaf(self):
        mock_cache, _, seg_d = self._construct_mock_cache()
        dual = mock_cache.get_dual()
        for i in range(seg_d[1], seg_d[15]):
            dual[i] = np.random.randn(dual[i].size).reshape(-1, 1)

        mock_cache.set_dual(dual)
        mock_cache.project_on_constraints_nonleaf()
        new_dual = mock_cache.get_dual()
        for i in range(len(dual)):
            self.assertTrue(new_dual[i].shape == dual[i].shape)

    def test_project_on_constraints_leaf(self):
        mock_cache, _, seg_d = self._construct_mock_cache()
        dual = mock_cache.get_dual()
        for i in range(seg_d[1], seg_d[15]):
            dual[i] = np.random.randn(dual[i].size).reshape(-1, 1)

        mock_cache.set_dual(dual)
        mock_cache.project_on_constraints_leaf()
        new_dual = mock_cache.get_dual()
        for i in range(len(dual)):
            self.assertTrue(new_dual[i].shape == dual[i].shape)

    def test_modify_projection(self):
        mock_cache, _, seg_d = self._construct_mock_cache()
        zero_dual = mock_cache.get_dual()
        parameter = 0
        while parameter == 0:
            parameter = np.linalg.norm(np.random.randn(1), 2)

        dual = mock_cache.get_dual()
        for i in range(seg_d[1], seg_d[15]):
            dual[i] = np.random.randn(dual[i].size).reshape(-1, 1)

        mock_cache.modify_projection(parameter, dual)
        new_dual = mock_cache.get_dual()
        for i in range(seg_d[1], seg_d[15]):
            test_result = parameter * (dual[i] - zero_dual[i])
            self.assertTrue(np.allclose(test_result, new_dual[i]))


if __name__ == '__main__':
    unittest.main()
