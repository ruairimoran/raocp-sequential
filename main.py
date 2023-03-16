import winsound
import pickle
import raocp as r
import numpy as np
import raocp.core.nodes as nodes
import raocp.core.dynamics as dynamics
import raocp.core.costs as costs
import raocp.core.risks as risks
import raocp.core.constraints.rectangle as rectangle
import raocp.core.printer as core_printer


def program_done():
    winsound.Beep(880, 500)


# ScenarioTree generation ----------------------------------------------------------------------------------------------

p = np.array([[0.1, 0.8, 0.1],
              [0.4, 0.6, 0.0],
              [0.0, 0.3, 0.7]])

v = np.array([0.1, 0.6, 0.3])

(N, tau) = (8, 3)
tree = r.core.MarkovChainScenarioTreeFactory(transition_prob=p,
                                             initial_distribution=v,
                                             num_stages=N, stopping_time=tau).create()

# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')
# print(sum(tree.probability_of_node(tree.nodes_at_stage(8))))
# print(tree)

# RAOCP generation -----------------------------------------------------------------------------------------------------
(nl, l) = nodes.Nonleaf(), nodes.Leaf()
(num_states, num_inputs) = 3, 2
factor = 0.1

Aw = factor * np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
Bw = factor * np.array([[3, 0], [1, 0], [0, 2]])
As = [1.5 * Aw, Aw, -1.5 * Aw]  # n x n matrices
Bs = [-1.5 * Bw, Bw, 1.5 * Bw]  # n x u matrices
mark_dynamics = [dynamics.Dynamics(As[0], Bs[0]),
                 dynamics.Dynamics(As[1], Bs[1]),
                 dynamics.Dynamics(As[2], Bs[2])]

Q = factor * np.eye(num_states)  # n x n matrix
Qs = [2 * Q, 2 * Q, 2 * Q]
R = factor * np.eye(num_inputs)  # u x u matrix OR scalar
Rs = [1 * R, 1 * R, 1 * R]
Pf = factor * 2 * np.eye(num_states)  # n x n matrix
mark_nl_costs = [costs.Quadratic(nl, Qs[0], Rs[0]),
                 costs.Quadratic(nl, Qs[1], Rs[1]),
                 costs.Quadratic(nl, Qs[2], Rs[2])]
leaf_cost = costs.Quadratic(l, Pf)

nonleaf_size = num_states + num_inputs
leaf_size = num_states
x_lim = 6
u_lim = 0.3
nl_min = np.vstack((-x_lim * np.ones((num_states, 1)),
                    -u_lim * np.ones((num_inputs, 1))))
nl_max = np.vstack((x_lim * np.ones((num_states, 1)),
                    u_lim * np.ones((num_inputs, 1))))
l_min = -x_lim * np.ones((leaf_size, 1))
l_max = x_lim * np.ones((leaf_size, 1))
nl_rect = rectangle.Rectangle(nl, nl_min, nl_max)
l_rect = rectangle.Rectangle(l, l_min, l_max)

alpha = .95
risk_1 = risks.AVaR(0.1)
risk_5 = risks.AVaR(0.5)
risk_9 = risks.AVaR(0.9)

problem_1 = r.core.RAOCP(scenario_tree=tree) \
    .with_markovian_dynamics(mark_dynamics) \
    .with_markovian_nonleaf_costs(mark_nl_costs) \
    .with_all_leaf_costs(leaf_cost) \
    .with_all_risks(risk_1) \
    .with_all_nonleaf_constraints(nl_rect) \
    .with_all_leaf_constraints(l_rect)

problem_5 = r.core.RAOCP(scenario_tree=tree) \
    .with_markovian_dynamics(mark_dynamics) \
    .with_markovian_nonleaf_costs(mark_nl_costs) \
    .with_all_leaf_costs(leaf_cost) \
    .with_all_risks(risk_5) \
    .with_all_nonleaf_constraints(nl_rect) \
    .with_all_leaf_constraints(l_rect)

problem_9 = r.core.RAOCP(scenario_tree=tree) \
    .with_markovian_dynamics(mark_dynamics) \
    .with_markovian_nonleaf_costs(mark_nl_costs) \
    .with_all_leaf_costs(leaf_cost) \
    .with_all_risks(risk_9) \
    .with_all_nonleaf_constraints(nl_rect) \
    .with_all_leaf_constraints(l_rect)

simple_solver_1 = r.core.Solver(problem_spec=problem_1, max_iters=10000, tol=1e-3)
simple_solver_5 = r.core.Solver(problem_spec=problem_5, max_iters=10000, tol=1e-3)
simple_solver_9 = r.core.Solver(problem_spec=problem_9, max_iters=10000, tol=1e-3)
super_solver_1 = r.core.Solver(problem_spec=problem_1, max_iters=10000, tol=1e-3)\
    .with_andersons_direction(memory_size=5)\
    .with_spock()
super_solver_5 = r.core.Solver(problem_spec=problem_5, max_iters=10000, tol=1e-3)\
    .with_andersons_direction(memory_size=5)\
    .with_spock()
super_solver_9 = r.core.Solver(problem_spec=problem_9, max_iters=10000, tol=1e-3)\
    .with_andersons_direction(memory_size=5)\
    .with_spock()

initial_state = np.array([[5], [-6], [-1]])


# write = create new solution, read = use last solution
read, write = 'rb', 'wb'
# solver = [solver, filename.pk, command, plot name]
solvers = [[simple_solver_1, 'cp_1.pk', read, "cp, a=0.1"],
           [simple_solver_5, 'cp_5.pk', read, "cp, a=0.5"],
           [simple_solver_9, 'cp_9.pk', read, "cp, a=0.9"],
           [super_solver_5, 'spock_5.pk', read, "spock, a=0.5"],
           [super_solver_9, 'spock_9.pk', read, "spock, a=0.9"],
           [super_solver_1, 'spock_1.pk', read, "spock, a=0.1"]]
num_solvers = len(solvers)
status = [None] * num_solvers
iters = [None] * num_solvers
chock_calls = [None] * num_solvers
for i in range(num_solvers):
    if solvers[i][2] == write:
        status[i], iters[i], chock_calls[i] = solvers[i][0].run(initial_state=initial_state)
        if status[i] == 0:
            print(f"{solvers[i][1]} solver success at iteration {iters[i]}")
        else:
            print(f"{solvers[i][1]} solver fail at iteration {iters[i]}")
        program_done()
        # solvers[i][0].plot_residuals(solvers[i][1])
        # solvers[i][0].plot_solution(solvers[i][1])
        # solvers[i][0].print_states()
        # solvers[i][0].print_inputs()
    with open(solvers[i][1], solvers[i][2]) as fi:
        if solvers[i][2] == write:
            pickle.dump(solvers[i][0], fi)
        else:
            solvers[i][0] = pickle.load(fi)


# # plot specific solvers
# swas_print = core_printer.Printer(super_with_ander_solver.get_cache)
# swas_print.plot_residuals("super")

# # plot different solver residual comparisons
# xis = [0]
# caches = [simple_solver.get_cache, super_with_ander_solver.get_cache]
# names = ["cp", "cp+sm+ander"]
# for xi in xis:
#     core_printer.plot_residual_comparisons(xi, caches, names)

# plot different alpha solver comparisons
for xi in [0]:
    core_printer.plot_residual_comparisons(xi, solvers)
