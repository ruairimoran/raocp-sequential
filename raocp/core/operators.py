import numpy as np
import raocp.core.cache as core_cache


class Operator:
    """
    Linear operators
    """

    def __init__(self, cache: core_cache.Cache):
        self.__raocp = cache.get_raocp()
        self.__num_nonleaf_nodes = self.__raocp.tree.num_nonleaf_nodes
        self.__num_nodes = self.__raocp.tree.num_nodes
        self.__segment_p = cache.get_primal_segments()
        self.__segment_d = cache.get_dual_segments()
        self.__template_p, self.__template_d = cache.get_primal_and_dual()

    def ell(self, input_primal, output_dual):
        # create sections for ease of access
        states = input_primal[self.__segment_p[1]: self.__segment_p[2]]
        controls = input_primal[self.__segment_p[2]: self.__segment_p[3]]
        dual_risk_y = input_primal[self.__segment_p[3]: self.__segment_p[4]]
        relaxation_tau = input_primal[self.__segment_p[4]: self.__segment_p[5]]
        relaxation_s = input_primal[self.__segment_p[5]: self.__segment_p[6]]
        for i in range(self.__num_nonleaf_nodes):
            children_of_i = self.__raocp.tree.children_of(i)
            output_dual[self.__segment_d[1] + i] = dual_risk_y[i]
            output_dual[self.__segment_d[2] + i] = relaxation_s[i] \
                - self.__raocp.risk_at_node(i).vector_b.T @ dual_risk_y[i]
            for j in children_of_i:
                output_dual[self.__segment_d[3] + j] = \
                    self.__raocp.nonleaf_cost_at_node(j).sqrt_state_weights @ states[i]
                output_dual[self.__segment_d[4] + j] = \
                    self.__raocp.nonleaf_cost_at_node(j).sqrt_control_weights @ controls[i]
                half_tau = (0.5 * relaxation_tau[j]).reshape(-1, 1)
                output_dual[self.__segment_d[5] + j] = half_tau
                output_dual[self.__segment_d[6] + j] = half_tau

            if self.__raocp.nonleaf_constraint_at_node(i).is_active:
                output_dual[self.__segment_d[7] + i] = \
                    self.__raocp.nonleaf_constraint_at_node(i).state_matrix @ states[i] + \
                    self.__raocp.nonleaf_constraint_at_node(i).control_matrix @ controls[i]

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            output_dual[self.__segment_d[11] + i] = \
                self.__raocp.leaf_cost_at_node(i).sqrt_state_weights @ states[i]
            half_s = (0.5 * relaxation_s[i]).reshape(-1, 1)
            output_dual[self.__segment_d[12] + i] = half_s
            output_dual[self.__segment_d[13] + i] = half_s
            if self.__raocp.leaf_constraint_at_node(i).is_active:
                output_dual[self.__segment_d[14] + i] = self.__raocp.leaf_constraint_at_node(i).state_matrix @ states[i]

    def ell_transpose(self, input_dual, output_primal):
        # create sections for ease of access
        dual_1 = input_dual[self.__segment_d[1]: self.__segment_d[2]]
        dual_2 = input_dual[self.__segment_d[2]: self.__segment_d[3]]
        dual_3 = input_dual[self.__segment_d[3]: self.__segment_d[4]]
        dual_4 = input_dual[self.__segment_d[4]: self.__segment_d[5]]
        dual_5 = input_dual[self.__segment_d[5]: self.__segment_d[6]]
        dual_6 = input_dual[self.__segment_d[6]: self.__segment_d[7]]
        dual_7 = input_dual[self.__segment_d[7]: self.__segment_d[8]]
        dual_11 = input_dual[self.__segment_d[11]: self.__segment_d[12]]
        dual_12 = input_dual[self.__segment_d[12]: self.__segment_d[13]]
        dual_13 = input_dual[self.__segment_d[13]: self.__segment_d[14]]
        dual_14 = input_dual[self.__segment_d[14]: self.__segment_d[15]]

        for i in range(self.__num_nonleaf_nodes):
            children_of_i = self.__raocp.tree.children_of(i)
            output_primal[self.__segment_p[3] + i] = \
                dual_1[i] - (self.__raocp.risk_at_node(i).vector_b @ dual_2[i]).reshape(-1, 1)
            output_primal[self.__segment_p[5] + i] = dual_2[i]
            if self.__raocp.nonleaf_constraint_at_node(i).is_active:
                output_primal[self.__segment_p[1] + i] = \
                    (self.__raocp.nonleaf_constraint_at_node(i).state_matrix_transposed @ dual_7[i]).reshape(-1, 1)
                output_primal[self.__segment_p[2] + i] = \
                    (self.__raocp.nonleaf_constraint_at_node(i).control_matrix_transposed @ dual_7[i]).reshape(-1, 1)
            else:
                output_primal[self.__segment_p[1] + i] = 0
                output_primal[self.__segment_p[2] + i] = 0
            for j in children_of_i:
                output_primal[self.__segment_p[1] + i] = output_primal[self.__segment_p[1] + i] + \
                    (self.__raocp.nonleaf_cost_at_node(j).sqrt_state_weights @ dual_3[j])
                output_primal[self.__segment_p[2] + i] = output_primal[self.__segment_p[2] + i] + \
                    (self.__raocp.nonleaf_cost_at_node(j).sqrt_control_weights @ dual_4[j])
                output_primal[self.__segment_p[4] + j] = 0.5 * (dual_5[j] + dual_6[j])

        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            output_primal[self.__segment_p[1] + i] = self.__raocp.leaf_cost_at_node(i).sqrt_state_weights @ dual_11[i]
            if self.__raocp.leaf_constraint_at_node(i).is_active:
                output_primal[self.__segment_p[1] + i] = output_primal[self.__segment_p[1] + i] + \
                    (self.__raocp.leaf_constraint_at_node(i).state_matrix_transposed @ dual_14[i]).reshape(-1, 1)
            output_primal[self.__segment_p[5] + i] = 0.5 * (dual_12[i] + dual_13[i])

    def linop_ell(self, flat_primal):
        prim = self.__template_p.copy()
        dual = self.__template_d.copy()
        cursor = 0
        for i in range(len(self.__template_p)):
            cursor_next = cursor + self.__template_p[i].size
            prim[i] = flat_primal[cursor: cursor_next].reshape(-1, 1)
            cursor = cursor_next

        self.ell(prim, dual)
        flat_dual = np.vstack(dual)
        return flat_dual

    def linop_ell_transpose(self, flat_dual):
        prim = self.__template_p.copy()
        dual = self.__template_d.copy()
        cursor = 0
        for i in range(len(self.__template_d)):
            cursor_next = cursor + self.__template_d[i].size
            dual[i] = flat_dual[cursor: cursor_next].reshape(-1, 1)
            cursor = cursor_next

        self.ell_transpose(dual, prim)
        flat_prim = np.vstack(prim)
        return flat_prim
