import numpy as np


class Direction:
    """
    Base class for storing direction type
    """

    @property
    def is_residuals(self):
        return False

    @property
    def is_andersons(self):
        return False


class Residuals(Direction):
    """
    Class for using the negative residual as a direction of descent
    """

    def __init__(self):
        self.__is_residuals = True

    @property
    def is_residuals(self):
        return True

    @staticmethod
    def get_direction(resid_zen):
        return -resid_zen


class Andersons(Direction):
    """
    Class for using Anderson's direction of descent
    """

    def __init__(self, memory_size=5):
        self.__andys_memory_size = memory_size
        self.__andys_point = []
        self.__andys_resid = []
        self.__andys_point_diff = []
        self.__andys_resid_diff = []
        self.__andys_point_diff_matrix = None
        self.__andys_resid_diff_matrix = None

    @property
    def is_andersons(self):
        return True

    def run_setup(self, zen_kplus1_, resid_zen_kplus1_, iplus1_):
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

    def get_direction(self, iplus1_):
        i_ = iplus1_ - 1
        # least squares
        gamma_k_ = np.linalg.lstsq(self.__andys_resid_diff_matrix, self.__andys_resid[i_], rcond=None)[0]
        direction_ = - self.__andys_resid[i_] \
                     - ((self.__andys_point_diff_matrix - self.__andys_resid_diff_matrix) @ gamma_k_)
        return direction_

    def update_buffer(self, zen_kplus1_, resid_zen_kplus1_, iplus2_):
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
