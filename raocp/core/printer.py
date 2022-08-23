import numpy as np
import raocp.core.solver as core_solver
import matplotlib.pyplot as plt
import tikzplotlib as tikz


def plot_residual_comparisons(xi, cache1, cache2, cache3, solver1_name, solver2_name, solver3_name):
    width = 2
    error1 = cache1.get_error_cache()
    error2 = cache2.get_error_cache()
    error3 = cache3.get_error_cache()
    plt.semilogy(error1[:, xi], linewidth=width, linestyle="solid")
    plt.semilogy(error2[:, xi], linewidth=width, linestyle="solid")
    plt.semilogy(error3[:, xi], linewidth=width, linestyle="solid")
    plt.title(f"comparison of {solver1_name}, {solver2_name} and {solver3_name} solvers xi_{xi} residual value")
    plt.ylabel(r"log(residual value)", fontsize=12)
    plt.xlabel(r"iteration", fontsize=12)
    plt.legend((f"{solver1_name}", f"{solver2_name}", f"{solver3_name}"))
    plt.show()


class Printer:
    """
    Printer and plotter for raocp solutions
    """

    def __init__(self, solver: core_solver.Solver):
        self.__solver = solver
        self.__cache = solver.get_cache()
        self.__error_cache = self.__cache.get_error_cache()

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

    def plot_residuals(self, solver_name):
        width = 2
        plt.semilogy(self.__error_cache[:, 0], linewidth=width, linestyle="solid")
        plt.semilogy(self.__error_cache[:, 1], linewidth=width, linestyle="solid")
        plt.semilogy(self.__error_cache[:, 2], linewidth=width, linestyle="solid")
        plt.title(f"{solver_name} solver residual values of Chambolle-Pock algorithm iterations")
        plt.ylabel(r"log(residual value)", fontsize=12)
        plt.xlabel(r"iteration", fontsize=12)
        plt.legend(("xi_0", "xi_1", "xi_2"))
        tikz.save('4-3-residuals.tex')
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
