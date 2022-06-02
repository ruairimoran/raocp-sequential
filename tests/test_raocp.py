import unittest
import raocp.core.scenario_tree as core_tree
import raocp.core.problem_spec as core_spec
import numpy as np


class TestRAOCP(unittest.TestCase):
    __tree_from_markov = None
    __tree_from_iid = None
    __raocp_from_markov = None
    __raocp_from_markov_with_markov = None
    __raocp_from_iid = None
    __good_size = 3

    @staticmethod
    def __construct_tree_from_markov():
        if TestRAOCP.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.5, 0])
            (N, tau) = (4, 3)
            TestRAOCP.__tree_from_markov = \
                core_tree.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

    @staticmethod
    def __construct_raocp_from_markov():
        if TestRAOCP.__raocp_from_markov is None:
            tree = TestRAOCP.__tree_from_markov

            # construct markovian set of system and control dynamics
            system = np.eye(2)
            set_system = [system, 2 * system, 3 * system]  # n x n matrices
            control = np.eye(2)
            set_control = [control, 2 * control, 3 * control]  # n x u matrices

            # construct cost weight matrices
            cost_type = "Quadratic"
            cost_types = [cost_type] * 3
            nonleaf_state_weight = 10 * np.eye(2)  # n x n matrix
            nonleaf_state_weights = [nonleaf_state_weight, 2 * nonleaf_state_weight, 3 * nonleaf_state_weight]
            control_weight = np.eye(2)  # u x u matrix OR scalar
            control_weights = [control_weight, 2 * control_weight, 3 * control_weight]
            leaf_state_weight = 5 * np.eye(2)  # n x n matrix

            # define risks
            (risk_type, alpha) = ("AVaR", 0.5)

            # create problem
            TestRAOCP.__raocp_from_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_all_nonleaf_costs(cost_type, nonleaf_state_weight, control_weight) \
                .with_all_leaf_costs(cost_type, leaf_state_weight) \
                .with_all_risks(risk_type, alpha)

            TestRAOCP.__raocp_from_markov_with_markov = core_spec.RAOCP(scenario_tree=tree) \
                .with_markovian_dynamics(set_system, set_control) \
                .with_markovian_costs(cost_types, nonleaf_state_weights, control_weights) \
                .with_all_leaf_costs(cost_type, leaf_state_weight) \
                .with_all_risks(risk_type, alpha)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestRAOCP.__construct_tree_from_markov()
        TestRAOCP.__construct_raocp_from_markov()

    def test_markovian_dynamics_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov_with_markov
        self.assertTrue(raocp.list_of_dynamics[0] is None)
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_dynamics[i] is not None)

    def test_markovian_nonleaf_costs_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov_with_markov
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_nonleaf_costs[i] is not None)

    def test_all_nonleaf_costs_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_nonleaf_costs[i] is not None)

    def test_leaf_costs_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i in range(tree.num_nodes):
            if i < tree.num_nonleaf_nodes:
                self.assertTrue(raocp.list_of_leaf_costs[i] is None)
            else:
                self.assertTrue(raocp.list_of_leaf_costs[i] is not None)

    def test_markovian_nonleaf_costs_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov_with_markov
        for i in range(1, tree.num_nodes):
            self.assertTrue(raocp.list_of_nonleaf_costs[i] is not None)

    def test_leaf_costs_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i in range(tree.num_nodes):
            if i < tree.num_nonleaf_nodes:
                self.assertTrue(raocp.list_of_leaf_costs[i] is None)
            else:
                self.assertTrue(raocp.list_of_leaf_costs[i] is not None)

    def test_risks_list(self):
        tree = TestRAOCP.__tree_from_markov
        raocp = TestRAOCP.__raocp_from_markov
        for i in range(tree.num_nonleaf_nodes):
            self.assertTrue(raocp.list_of_risks[i] is not None)

    def test_markovian_system_dynamics_failure(self):
        tree = TestRAOCP.__tree_from_markov

        # construct bad markovian set of system dynamics
        set_system = [np.eye(TestRAOCP.__good_size + 1), np.eye(TestRAOCP.__good_size)]  # n x n matrices
        # construct good markovian set of control dynamics
        control = np.ones((TestRAOCP.__good_size, 1))
        set_control = [control, 2 * control]  # n x u matrices

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_markovian_dynamics(set_system, set_control)

    def test_markovian_control_dynamics_failure(self):
        tree = TestRAOCP.__tree_from_markov

        # construct good markovian set of system dynamics
        system = np.eye(TestRAOCP.__good_size)
        set_system = [system, 2 * system]  # n x n matrices
        # construct bad markovian set of control dynamics
        set_control = [np.ones((TestRAOCP.__good_size + 1, 1)), np.ones((TestRAOCP.__good_size, 1))]  # n x u matrices

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_markovian_dynamics(set_system, set_control)

    def test_markovian_system_and_control_dynamics_failure(self):
        tree = TestRAOCP.__tree_from_markov

        # construct good markovian set of system dynamics (rows = 3)
        system = np.eye(TestRAOCP.__good_size)
        set_system = [system, 2 * system]  # n x n matrices
        # construct good markovian set of control dynamics  (rows = 4)
        control = np.ones((TestRAOCP.__good_size + 1, 1))
        set_control = [control, 2 * control]  # n x u matrices

        # construct problem with error catch
        with self.assertRaises(ValueError):
            _ = core_spec.RAOCP(tree).with_markovian_dynamics(set_system, set_control)


if __name__ == '__main__':
    unittest.main()
