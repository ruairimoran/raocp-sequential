import unittest
import numpy as np
import raocp.core.cache as core_cache
import raocp.core.costs as core_costs
import raocp.core.dynamics as core_dynamics
import raocp.core.nodes as core_nodes
import raocp.core.operators as core_operators
import raocp.core.raocp_spec as core_spec
import raocp.core.risks as core_risks
import raocp.core.scenario_tree as core_tree


class TestOperators(unittest.TestCase):
    __tree_from_markov = None
    __raocp_from_markov = None
    __cache_from_raocp = None
    __operators_from_cache = None
    __good_size = 3

    @staticmethod
    def _construct_tree_from_markov():
        if TestOperators.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.4, 0.1])
            (N, tau) = (4, 3)
            TestOperators.__tree_from_markov = core_tree.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

    @staticmethod
    def _construct_raocp_from_markov():
        if TestOperators.__raocp_from_markov is None:
            tree = TestOperators.__tree_from_markov

            # construct markovian set of system and control dynamics
            system = np.eye(2)
            set_system = [system, 2 * system, 3 * system]  # n x n matrices
            control = np.eye(2)
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
                             core_costs.Quadratic(nl, nonleaf_state_weights[2], control_weights[2])]
            leaf_state_weight = 5 * np.eye(2)  # n x n matrix
            leaf_costs = core_costs.Quadratic(l, leaf_state_weight)

            # define risks
            alpha = 0.5
            risks = core_risks.AVaR(alpha)

            # create problem
            TestOperators.__raocp_from_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_markovian_dynamics(dynamics) \
                .with_markovian_nonleaf_costs(nonleaf_costs) \
                .with_all_leaf_costs(leaf_costs) \
                .with_all_risks(risks)

    @staticmethod
    def _construct_cache_from_raocp():
        if TestOperators.__cache_from_raocp is None:
            TestOperators.__cache_from_raocp = core_cache.Cache(TestOperators.__raocp_from_markov)

    @staticmethod
    def _construct_operators_from_cache():
        if TestOperators.__operators_from_cache is None:
            TestOperators.__operators_from_cache = core_operators.Operator(TestOperators.__cache_from_raocp)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestOperators._construct_tree_from_markov()
        TestOperators._construct_raocp_from_markov()
        TestOperators._construct_cache_from_raocp()
        TestOperators._construct_operators_from_cache()

    @staticmethod
    def setup():
        # get primal and dual segments
        segments_p = TestOperators.__cache_from_raocp.get_primal_segments()
        segments_d = TestOperators.__cache_from_raocp.get_dual_segments()
        # get template of primal and dual for sizes
        primal = TestOperators.__cache_from_raocp.get_primal()
        dual = TestOperators.__cache_from_raocp.get_dual()
        # setup memory
        random_primal = primal.copy()
        random_dual = dual.copy()
        ell_transpose_dual = primal.copy()
        ell_primal = dual.copy()
        return segments_p, segments_d, random_primal, random_dual, ell_transpose_dual, ell_primal

    @staticmethod
    def inner_product(random_primal, ell_transpose_dual, random_dual, ell_primal):
        # get ell and ell_transpose
        TestOperators.__operators_from_cache.ell(random_primal, ell_primal)
        TestOperators.__operators_from_cache.ell_transpose(random_dual, ell_transpose_dual)

        # get inner products - np.inner takes two row vectors to give a scalar (transposes second argument)
        inner_primal = 0
        for i in range(len(random_primal)):
            inner_primal += np.inner(random_primal[i].T, ell_transpose_dual[i].T)

        inner_dual = 0
        for i in range(len(random_dual)):
            inner_dual += np.inner(ell_primal[i].T, random_dual[i].T)

        return inner_primal, inner_dual

    def test_y_and_1(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        y = 3
        for i in range(seg_prim[y], seg_prim[y+1]):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 1
        for i in range(seg_dual[eta], seg_dual[eta+1]):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_s_nonleaf_and_2(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        s = 5
        for i in range(seg_prim[s] + 1, seg_prim[s] + self.__tree_from_markov.num_nonleaf_nodes):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 2
        for i in range(seg_dual[eta] + 1, seg_dual[eta] + self.__tree_from_markov.num_nonleaf_nodes):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_x_nonleaf_and_3(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        x = 1
        for i in range(seg_prim[x], seg_prim[x] + self.__tree_from_markov.num_nonleaf_nodes):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 3
        for i in range(seg_dual[eta], seg_dual[eta + 1]):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_u_and_4(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        u = 2
        for i in range(seg_prim[u], seg_prim[u] + self.__tree_from_markov.num_nonleaf_nodes):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 4
        for i in range(seg_dual[eta], seg_dual[eta + 1]):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_t_and_5(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        t = 4
        for i in range(seg_prim[t] + 1, seg_prim[t + 1]):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 5
        for i in range(seg_dual[eta] + 1, seg_dual[eta + 1]):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_t_and_6(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        t = 4
        for i in range(seg_prim[t] + 1, seg_prim[t + 1]):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 6
        for i in range(seg_dual[eta] + 1, seg_dual[eta + 1]):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_x_nonleaf_and_7(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        x = 1
        for i in range(seg_prim[x], seg_prim[x] + self.__tree_from_markov.num_nonleaf_nodes):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 7
        for i in range(seg_dual[eta], seg_dual[eta + 1]):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_u_and_7(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        u = 2
        for i in range(seg_prim[u], seg_prim[u + 1]):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 7
        for i in range(seg_dual[eta], seg_dual[eta + 1]):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_x_leaf_and_11(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        x = 1
        for i in range(seg_prim[x] + self.__tree_from_markov.num_nonleaf_nodes, seg_prim[x + 1]):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 11
        for i in range(seg_dual[eta] + self.__tree_from_markov.num_nonleaf_nodes, seg_dual[eta + 1]):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_s_leaf_and_12(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        s = 5
        for i in range(seg_prim[s] + self.__tree_from_markov.num_nonleaf_nodes, seg_prim[s + 1]):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        peta = [12, 13]
        for eta in peta:
            for i in range(seg_dual[eta] + self.__tree_from_markov.num_nonleaf_nodes, seg_dual[eta + 1]):
                rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        # get ell and ell_transpose
        TestOperators.__operators_from_cache.ell(rand_prim, ell_prim)
        TestOperators.__operators_from_cache.ell_transpose(rand_dual, ell_t_dual)

        # get inner products - np.inner takes two row vectors to give a scalar (transposes second argument)
        inner_prim = 0
        for i in range(len(rand_prim)):
            inner_prim += np.inner(rand_prim[i].T, ell_t_dual[i].T)

        inner_dual = 0
        for i in range(len(rand_dual)):
            inner_dual += np.inner(ell_prim[i].T, rand_dual[i].T)

        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_s_leaf_and_13(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        s = 5
        for i in range(seg_prim[s] + self.__tree_from_markov.num_nonleaf_nodes, seg_prim[s + 1]):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 13
        for i in range(seg_dual[eta] + self.__tree_from_markov.num_nonleaf_nodes, seg_dual[eta + 1]):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_x_leaf_and_14(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        x = 1
        for i in range(seg_prim[x] + self.__tree_from_markov.num_nonleaf_nodes, seg_prim[x + 1]):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        eta = 14
        for i in range(seg_dual[eta] + self.__tree_from_markov.num_nonleaf_nodes, seg_dual[eta + 1]):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_operators(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        for i in range(len(rand_prim)):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # create random values for dual
        for i in range(len(rand_dual)):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        inner_prim, inner_dual = TestOperators.inner_product(rand_prim, ell_t_dual, rand_dual, ell_prim)
        self.assertAlmostEqual(inner_prim[0, 0], inner_dual[0, 0])

    def test_linop_ell(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for primal
        for i in range(len(rand_prim)):
            rand_prim[i] = np.random.randn(rand_prim[i].size).reshape((-1, 1))

        # flatten primal
        flat_rand_prim = np.vstack(rand_prim)

        # get dual from ell and flatten
        TestOperators.__operators_from_cache.ell(rand_prim, ell_prim)
        flat_unwrapped = np.vstack(ell_prim)

        # get dual from ell wrapper
        flat_wrapped = TestOperators.__operators_from_cache.linop_ell(flat_rand_prim)

        self.assertTrue(np.array_equal(flat_unwrapped, flat_wrapped))

    def test_linop_ell_transpose(self):
        seg_prim, seg_dual, rand_prim, rand_dual, ell_t_dual, ell_prim = TestOperators.setup()

        # create random values for dual
        for i in range(len(rand_dual)):
            rand_dual[i] = np.random.randn(rand_dual[i].size).reshape((-1, 1))

        # flatten dual
        flat_rand_dual = np.vstack(rand_dual)

        # get primal from ell_transpose and flatten
        TestOperators.__operators_from_cache.ell_transpose(rand_dual, ell_t_dual)
        flat_unwrapped = np.vstack(ell_t_dual)

        # get primal from ell_transpose wrapper
        flat_wrapped = TestOperators.__operators_from_cache.linop_ell_transpose(flat_rand_dual)

        self.assertTrue(np.array_equal(flat_unwrapped, flat_wrapped))


if __name__ == '__main__':
    unittest.main()
