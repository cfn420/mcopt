# feasibility.py

import logging
import numpy as np
from network import x_to_matrix

logger = logging.getLogger(__name__)

def feasible(problem, x, tol=1e-8, return_P=False):
    """
    Check that a candidate decision vector x yields a valid transition matrix P.

    Parameters:
      - problem (ProblemInstance): The problem instance with attributes:
          N (int): number of states
          edge_matrix (list): edges mapping used by x_to_matrix
          bUndirected (bool): whether edges are undirected
          bPi_constraint (bool): whether a pi_hat constraint is active
          pi_hat (np.ndarray): target stationary distribution (if bPi_constraint)
          eta (float): lower-bound margin for x
      - x (np.ndarray): 1D decision vector of length problem.d
      - tol (float): absolute tolerance for numerical checks
      - return_P (bool): if True, return the reconstructed P matrix

    Returns:
      - np.ndarray: The N×N transition matrix P if return_P is True
      - None: otherwise

    Raises:
      - ValueError: if any feasibility check fails
    """
    N = problem.N
    P = x_to_matrix(x, N, problem.edge_matrix, problem.bUndirected)

    # Check that each row sums to 1
    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, np.ones(N), atol=tol):
        logger.error("Row sums incorrect for P: %s", row_sums)
        raise ValueError(f"Row sums incorrect (tol={tol}): {row_sums}")

    # Stationary distribution constraint, if present
    if getattr(problem, 'bPi_constraint', False):
        lhs = problem.pi_hat @ P
        if not np.allclose(lhs, problem.pi_hat, atol=tol):
            logger.error(
                "Stationary distribution violated: pi_hat*P=%s, pi_hat=%s",
                lhs,
                problem.pi_hat,
            )
            raise ValueError("Stationary distribution constraint violated.")

    # Bounds on decision vector x
    x_min = np.min(x)
    if x_min < (problem.eta - tol):
        logger.error(
            "Decision vector x below lower bound: min(x)=%s, eta=%s", x_min, problem.eta
        )
        raise ValueError(f"Elements of x below lower bound (eta={problem.eta}).")

    x_max = np.max(x)
    if x_max > (1 - problem.eta + tol):
        logger.error(
            "Decision vector x above upper bound: max(x)=%s, 1-eta=%s", x_max, 1 - problem.eta
        )
        raise ValueError(f"Elements of x above upper bound (1-eta={1-problem.eta}).")

    if return_P:
        return P