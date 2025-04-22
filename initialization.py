# initialization.py

import numpy as np

from optimization.projections import projection
from optimization.feasibility import feasible
from network import MarkovChain

def sample_single_initial(problem):
    """
    Samples a single feasible initial solution using projection and feasibility check.

    Parameters:
        problem (ProblemInstance): Problem instance with projection matrices ready.

    Returns:
        np.ndarray: A single feasible solution vector x.
    """
    x = np.random.uniform(size=problem.d)
    x = projection(x, problem, bnds=(problem.eta, 1 - problem.eta), tol=1e-8)
    feasible(problem, x)  # Raises a warning if not feasible

    return x

def sample_initials(problem, size):
    """
    Samples multiple initial feasible solutions.

    Parameters:
        problem (ProblemInstance): Problem instance with projection matrices ready.
        size (int): Number of samples to generate.

    Returns:
        np.ndarray: A (size x d) matrix of feasible initial solutions.
    """
    N, N_check = problem.mA.shape
    if N != N_check:
        raise ValueError("Matrix mA must be square (N x N).")

    d = problem.A.shape[1]
    vX = np.zeros((size, d))

    for i in range(size):
        vX[i] = sample_single_initial(problem)

    return vX

def sample_best(problem, n_samples=10):
    """
    Samples multiple initial feasible solutions and returns the best one
    based on the problem objective evaluated through the MarkovChain instance.

    Parameters:
        problem (ProblemInstance): Problem instance with objective and matrices.
        n_samples (int): Number of samples to draw.

    Returns:
        mc (MarkovChain): with its .x set to the best sample.
    """
    vX = sample_initials(problem, n_samples)
    mc = MarkovChain(mA=problem.mA, bUndirected=problem.bUndirected)  # Initialize once, x is set inside loop

    scores = []
    for x in vX:
        mc.x = x  # Update x in the MarkovChain instance

        score = problem.objective(mc.P_matrix)
        scores.append(score)

    scores = np.array(scores)
    best_idx = np.argmin(scores) 
    best_x = np.copy(vX[best_idx])
    mc = MarkovChain(mA=problem.mA, x=np.copy(best_x), bUndirected=problem.bUndirected)  # Finalize with best x

    return mc

