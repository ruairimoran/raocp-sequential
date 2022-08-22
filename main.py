import winsound
import pickle
import raocp as r
import numpy as np
import raocp.core.nodes as nodes
import raocp.core.dynamics as dynamics
import raocp.core.costs as costs
import raocp.core.risks as risks
import raocp.core.constraints.rectangle as rectangle


def program_done():
    winsound.Beep(880, 500)


# ScenarioTree generation ----------------------------------------------------------------------------------------------

p = np.array([[0.1, 0.8, 0.1],
              [0.4, 0.6, 0.0],
              [0.0, 0.3, 0.7]])

v = np.array([0.1, 0.6, 0.3])

(N, tau) = (6, 2)
tree = r.core.MarkovChainScenarioTreeFactory(transition_prob=p,
                                             initial_distribution=v,
                                             num_stages=N, stopping_time=tau).create()

# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')
# print(sum(tree.probability_of_node(tree.nodes_at_stage(8))))
# print(tree)

# RAOCP generation -----------------------------------------------------------------------------------------------------
(nl, l) = nodes.Nonleaf(), nodes.Leaf()
(num_states, num_inputs) = 3, 2
factor = .05

Aw = factor * np.array([[1, 2, 1], [1, 1, 2], [2, 1, 1]])
Bw = factor * np.array([[1, 0], [1, 2], [0, 2]])
As = [0.5 * Aw, Aw, -0.5 * Aw]  # n x n matrices
Bs = [-0.5 * Bw, Bw, 0.5 * Bw]  # n x u matrices
mark_dynamics = [dynamics.Dynamics(As[0], Bs[0]),
                 dynamics.Dynamics(As[1], Bs[1]),
                 dynamics.Dynamics(As[2], Bs[2])]

Q = factor * np.eye(num_states)  # n x n matrix
Qs = [.2 * Q, .2 * Q, .2 * Q]
R = factor * np.eye(num_inputs)  # u x u matrix OR scalar
Rs = [.2 * R, .2 * R, .2 * R]
Pf = factor * 50 * np.eye(num_states)  # n x n matrix
mark_nl_costs = [costs.Quadratic(nl, Qs[0], Rs[0]),
                 costs.Quadratic(nl, Qs[1], Rs[1]),
                 costs.Quadratic(nl, Qs[2], Rs[2])]
leaf_cost = costs.Quadratic(l, Pf)

nonleaf_size = num_states + num_inputs
leaf_size = num_states
x_lim = 7
u_lim = 0.01
nl_min = np.vstack((-x_lim * np.ones((num_states, 1)),
                    -u_lim * np.ones((num_inputs, 1))))
nl_max = np.vstack((x_lim * np.ones((num_states, 1)),
                    u_lim * np.ones((num_inputs, 1))))
l_min = -x_lim * np.ones((leaf_size, 1))
l_max = x_lim * np.ones((leaf_size, 1))
nl_rect = rectangle.Rectangle(nl, nl_min, nl_max)
l_rect = rectangle.Rectangle(l, l_min, l_max)

alpha = .95
risk = risks.AVaR(alpha)

problem = r.core.RAOCP(scenario_tree=tree) \
    .with_markovian_dynamics(mark_dynamics) \
    .with_markovian_nonleaf_costs(mark_nl_costs) \
    .with_all_leaf_costs(leaf_cost) \
    .with_all_risks(risk) \
    .with_all_nonleaf_constraints(nl_rect) \
    .with_all_leaf_constraints(l_rect)

simple_solver = r.core.Solver(problem_spec=problem, max_outer_iters=5000, tol=1e-3)
super_solver = r.core.Solver(problem_spec=problem, max_outer_iters=5000, tol=1e-3)
resid_solver = r.core.Solver(problem_spec=problem, max_outer_iters=5000, tol=1e-3)
initial_state = np.array([[5], [-6], [-1]])


# write = create new solution, read = use last solution
read, write = 'rb', 'wb'
filename = ['simple_solver.pk', 'super_solver.pk', 'resid_solver.pk']
command = [read, read, read]


# simple chock
if command[0] == write:
    simple_chock_status, iters = simple_solver.simple_chock(initial_state=initial_state)
    if simple_chock_status == 0:
        print(f"simple chock success at iteration {iters}")
    else:
        print(f"simple chock fail at iteration {iters}")
    program_done()
    # simple_solver.plot_residuals("simple")
    # simple_solver.plot_solution("simple")
    # simple_solver.print_states()
    # simple_solver.print_inputs()


# super chock
if command[1] == write:
    super_chock_status, outer_iters, chock_calls = super_solver.super_chock(initial_state=initial_state)
    if super_chock_status == 0:
        print(f"super chock success: outer = {outer_iters}, chock calls = {chock_calls}")
    else:
        print(f"super chock fail: outer = {outer_iters}, chock calls = {chock_calls}")
    program_done()
    # super_solver.plot_residuals("super")
    # super_solver.plot_solution("super")
    # super_solver.print_states()
    # super_solver.print_inputs()


# resid chock
if command[2] == write:
    resid_chock_status, outer_iters, chock_calls = resid_solver.super_chock(initial_state=initial_state,
                                                                            andersons_setup_iterations=1000)
    if resid_chock_status == 0:
        print(f"resid chock success: outer = {outer_iters}, chock calls = {chock_calls}")
    else:
        print(f"resid chock fail: outer = {outer_iters}, chock calls = {chock_calls}")
    program_done()
    # super_solver.plot_residuals("super")
    # super_solver.plot_solution("super")
    # super_solver.print_states()
    # super_solver.print_inputs()


# use pickle to store solved problems instead of rerunning
with open(filename[0], command[0]) as fi:
    if command[0] == write:
        pickle.dump(simple_solver, fi)
    else:
        simple_solver = pickle.load(fi)
with open(filename[1], command[1]) as fi:
    if command[1] == write:
        pickle.dump(super_solver, fi)
    else:
        super_solver = pickle.load(fi)
with open(filename[2], command[2]) as fi:
    if command[2] == write:
        pickle.dump(resid_solver, fi)
    else:
        resid_solver = pickle.load(fi)


# plot comparisons
for xi in [0]:  # , 1, 2]:
    simple_solver.plot_residual_comparisons(xi, simple_solver, super_solver, resid_solver, "simple", "super", "resid")
