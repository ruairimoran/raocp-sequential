import numpy as np
from scipy.linalg import sqrtm
import raocp.core.problem_spec as ps


class Operator:
    """
    Linear operators
    """

    def __init__(self, problem_spec: ps.RAOCP, primal_split, dual_split):
        self.__raocp = problem_spec
        self.__num_nonleaf_nodes = self.__raocp.tree.num_nonleaf_nodes
        self.__num_nodes = self.__raocp.tree.num_nodes
        self.__primal_split = primal_split
        self.__primal = [] * self.__primal_split[-1]
        self.__dual_split = dual_split
        self.__dual = [] * self.__dual_split[-1]
        self._create_sections()

    def _create_sections(self):
        # primal
        self.__states = self.__primal[self.__primal_split[0]: self.__primal_split[1]]  # x
        self.__controls = self.__primal[self.__primal_split[1]: self.__primal_split[2]]  # u
        self.__dual_risk_y = self.__primal[self.__primal_split[2]: self.__primal_split[3]]  # y
        self.__relaxation_s = self.__primal[self.__primal_split[3]: self.__primal_split[4]]  # s
        self.__relaxation_tau = self.__primal[self.__primal_split[4]: self.__primal_split[5]]  # tau
        # dual
        self.__dual_1 = self.__dual[self.__dual_split[0]: self.__dual_split[1]]
        self.__dual_2 = self.__dual[self.__dual_split[1]: self.__dual_split[2]]
        self.__dual_3 = self.__dual[self.__dual_split[2]: self.__dual_split[3]]
        self.__dual_4 = self.__dual[self.__dual_split[3]: self.__dual_split[4]]
        self.__dual_5 = self.__dual[self.__dual_split[4]: self.__dual_split[5]]
        self.__dual_6 = self.__dual[self.__dual_split[5]: self.__dual_split[6]]
        self.__dual_7 = self.__dual[self.__dual_split[6]: self.__dual_split[7]]
        self.__dual_8 = self.__dual[self.__dual_split[7]: self.__dual_split[8]]
        self.__dual_9 = self.__dual[self.__dual_split[8]: self.__dual_split[9]]

    def ell(self, modified_primal):
        self.__primal = modified_primal
        for i in range(self.__num_nonleaf_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            stage_at_children_of_i = self.__raocp.tree.stage_of(i) + 1
            children_of_i = self.__raocp.tree.children_of(i)
            self.__dual_1[i] = self.__dual_risk_y[i]
            self.__dual_2[i] = self.__relaxation_s[stage_at_i][i] \
                - self.__raocp.risk_at_node(i).vector_b.T @ self.__dual_risk_y[i]
            for j in children_of_i:
                self.__dual_3[j] = sqrtm(self.__raocp.nonleaf_cost_at_node(j).nonleaf_state_weights) @ self.__states[i]
                self.__dual_4[j] = sqrtm(self.__raocp.nonleaf_cost_at_node(j).control_weights) @ self.__controls[i]
                half_tau = 0.5 * self.__relaxation_tau[stage_at_children_of_i][j]
                self.__dual_5[j] = half_tau
                self.__dual_6[j] = half_tau

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            self.__dual_7[i] = sqrtm(self.__raocp.leaf_cost_at_node(i).leaf_state_weights) @ self.__states[i]
            half_s = 0.5 * self.__relaxation_s[stage_at_i][i]
            self.__dual_8[i] = half_s
            self.__dual_9[i] = half_s

        return self.__dual

    def ell_transpose(self, modified_dual):
        self.__dual = modified_dual
        for i in range(self.__num_nonleaf_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            stage_at_children_of_i = self.__raocp.tree.stage_of(i) + 1
            children_of_i = self.__raocp.tree.children_of(i)
            self.__dual_risk_y[i] = (self.__dual_1[i] - self.__raocp.risk_at_node(i).vector_b @ self.__dual_2[i]) \
                .reshape((2 * self.__raocp.tree.children_of(i).size + 1, 1))  # reshape to column vector
            self.__relaxation_s[stage_at_i][i] = self.__dual_2[i]
            self.__states[i] = 0
            self.__controls[i] = 0
            for j in children_of_i:
                self.__states[i] += sqrtm(self.__raocp.nonleaf_cost_at_node(j).nonleaf_state_weights).T \
                                    @ self.__dual_3[j]
                self.__controls[i] += sqrtm(self.__raocp.nonleaf_cost_at_node(j).control_weights).T @ self.__dual_4[j]
                self.__relaxation_tau[stage_at_children_of_i][j] = 0.5 * (self.__dual_5[j] + self.__dual_6[j])

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            stage_at_i = self.__raocp.tree.stage_of(i)
            self.__states[i] = sqrtm(self.__raocp.leaf_cost_at_node(i).leaf_state_weights).T @ self.__dual_7[i]
            self.__relaxation_s[stage_at_i][i] = 0.5 * (self.__dual_8[i] + self.__dual_9[i])

        return self.__primal