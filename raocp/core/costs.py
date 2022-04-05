import numpy as np


class QuadraticNonleaf:
    """
    A quadratic cost item for any nonleaf node
    """

    def __init__(self, nonleaf_state_weights, control_weights):
        """
        :param nonleaf_state_weights: nonleaf node state cost matrix (Q)
        :param control_weights: input cost matrix (R)
        """
        if nonleaf_state_weights.shape[0] != nonleaf_state_weights.shape[1]:
            raise ValueError("quadratic cost state weight matrix is not square")
        else:
            self.__nonleaf_state_weights = nonleaf_state_weights
        self.__control_weights = control_weights
        self.__most_recent_cost_value = None

    def get_cost_value(self, state, control):
        """For calculating nonleaf cost"""
        if state.shape[0] != self.__nonleaf_state_weights.shape[0]:
            raise ValueError("quadratic cost input nonleaf state dimension does not match state weight matrix")
        if control.shape[0] != self.__control_weights.shape[0]:
            raise ValueError("quadratic cost input control dimension does not match control weight matrix")
        self.__most_recent_cost_value = state.T @ self.__nonleaf_state_weights @ state \
            + control.T @ self.__control_weights @ control
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def nonleaf_state_weights(self):
        return self.__nonleaf_state_weights

    @property
    def control_weights(self):
        return self.__control_weights

    @property
    def most_recent_cost_value(self):
        return self.__most_recent_cost_value[0, 0]

    def __str__(self):
        return f"Cost item; type: {type(self).__name__}"

    def __repr__(self):
        return f"Cost item; type: {type(self).__name__}"


class QuadraticLeaf:
    """
    A quadratic cost item for any leaf node
    """

    def __init__(self, leaf_state_weights):
        """
        :param leaf_state_weights: leaf node state cost matrix (Pf)
        """
        if leaf_state_weights.shape[0] != leaf_state_weights.shape[1]:
            raise ValueError("quadratic cost state weight matrix is not square")
        else:
            self.__leaf_state_weights = leaf_state_weights
        self.__most_recent_cost_value = None

    def get_cost_value(self, state):
        """For calculating leaf cost"""
        if state.shape[0] != self.__leaf_state_weights.shape[0]:
            raise ValueError("quadratic cost input leaf state dimension does not match state weight matrix")
        self.__most_recent_cost_value = state.T @ self.__leaf_state_weights @ state
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def leaf_state_weights(self):
        return self.__leaf_state_weights

    @property
    def most_recent_cost_value(self):
        return self.__most_recent_cost_value[0, 0]

    def __str__(self):
        return f"Cost item; type: {type(self).__name__}"

    def __repr__(self):
        return f"Cost item; type: {type(self).__name__}"
