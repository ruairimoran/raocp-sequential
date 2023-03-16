import numpy as np
import raocp.core.cache as core_cache
import matplotlib.pyplot as plt
import tikzplotlib as tikz


def plot_residual_comparisons(xi, solvers):
    width = 2
    num_solvers = len(solvers)
    error_caches = [None] * num_solvers
    plot_names = [None] * num_solvers
    for i in range(num_solvers):
        error_caches[i] = solvers[i][0].get_cache.get_error_cache()
        plt.semilogy(error_caches[i][:, xi], linewidth=width, linestyle="solid")
        plot_names[i] = solvers[i][3]

    plt.title(f"Comparison of solvers FPR, xi_{xi}")
    plt.ylabel(r"log(infinity-norm of FPR)", fontsize=12)
    plt.xlabel(r"iteration", fontsize=12)
    plt.legend(plot_names)
    tikz.save('8-3-comparison.tex')
    plt.show()


class Printer:
    """
    Printer and plotter for raocp solutions
    """

    def __init__(self, solver_cache: core_cache.Cache):
        self.__cache = solver_cache
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
