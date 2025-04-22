# projections.py

import numpy as np
from itertools import chain

from optimization.feasibility import feasible

def projection_box(x, bnds, tol=1e-8):
    """
    Project a vector onto a box defined by lower and upper bounds.

    Parameters:
        x (array_like): Input vector.
        bnds (tuple): Tuple of (lower_bound, upper_bound).
        tol (float): Unused here but reserved for interface consistency.

    Returns:
        np.ndarray: Vector clipped to the specified bounds.
    """
    return np.clip(x, bnds[0], bnds[1])

def projection_subspace(x, A_pinv_b, C__C_T_C_inv__C_T):
    """
    Project a vector x onto the affine subspace defined by Ax = b.

    Parameters:
        x (np.ndarray): Input vector.
        A_pinv_b (np.ndarray): Precomputed pseudo-inverse of A times b.
        C__C_T_C_inv__C_T (np.ndarray): Precomputed projection matrix for the nullspace.

    Returns:
        np.ndarray: The projection of x onto the subspace Ax = b.
    """
    y = x_to_y(x, A_pinv_b)
    return A_pinv_b + C__C_T_C_inv__C_T @ y

def proj_c_simplex_held(v, c=1, tol=1e-8):
    """
    Project a vector v onto the scaled probability simplex {x ≥ 0, sum(x) = c}.

    Parameters:
        v (np.ndarray): Input vector.
        c (float): Desired sum of projected vector (default is 1).
        tol (float): Tolerance for projection accuracy.

    Returns:
        np.ndarray: Projected vector on the simplex.

    Raises:
        ValueError: If a valid projection index cannot be found.
        Warning: If projection is inaccurate or produces negative values.
    """
    N = len(v)
    vU = np.sort(v)[::-1]
    cssv = np.cumsum(vU)
    l = [k+1 for k in range(N) if (cssv[k] - c) / (k + 1) < vU[k]]

    if not l: raise ValueError("No valid projection index found")

    K = max(l)
    tau = (cssv[K - 1] - c) / K
    v_proj = np.maximum(v - tau, 0)

    if abs(np.sum(v_proj) - c) > tol:
        raise Warning("Projection insufficiently accurate. Try reducing the step size.")
    if np.min(v_proj) < 0:
        raise Warning("Negative elements found.")

    return v_proj

def projection_markov(x_to_proj, bnds, neighborhoods, mA):
    """
    Project a vector onto a Markov-type local simplex for each state.

    Parameters:
        x_to_proj (np.ndarray): Input vector to project.
        bnds (tuple of float): (eta, 1 - eta) bounds. Only eta is used.
        neighborhoods (list of list[int]): Index subsets for local projections.
        mA (np.ndarray): Adjacency matrix used to compute mass for each row.

    Returns:
        np.ndarray: Projected vector satisfying local Markov constraints.
    """

    eta = bnds[0]
    x = x_to_proj - eta

    x_proj = [
        proj_c_simplex_held(x[subset], c=1 - np.sum(mA[i]) * eta).tolist()
        if subset else []
        for i, subset in enumerate(neighborhoods)
    ]

    return np.array(list(chain.from_iterable(x_proj))) + eta

def dykstra(x0, tol, A_pinv_b, C__C_T_C_inv__C_T, bnds=(1e-6, 1 - 1e-6), max_iter=1_000_000):
    """
    Apply Dykstra's algorithm to project onto an intersection of affine and box constraints.

    Parameters:
        x0 (np.ndarray): Initial vector.
        tol (float): Tolerance for convergence.
        A_pinv_b (np.ndarray): Precomputed pseudo-inverse of A times b.
        C__C_T_C_inv__C_T (np.ndarray): Projection matrix for nullspace component.
        bnds (tuple): Bounds for box projection.
        max_iter (int): Maximum number of iterations.

    Returns:
        np.ndarray: Projected vector satisfying both constraints.

    Raises:
        RuntimeError: If convergence fails.
        Warning: If projection yields invalid (negative) values.
    """
    x = x0.copy()
    p = np.zeros_like(x)
    q = np.zeros_like(x)

    for k in range(max_iter):
        z = projection_box(x + p, bnds)
        p += x - z
        x = projection_subspace(z + q, A_pinv_b, C__C_T_C_inv__C_T)
        q += z - x

        if np.min(x) > (bnds[0] - tol):
            break
    else:
        raise RuntimeError("Projection failed to converge.")

    if np.min(x) < 0:
        raise Warning("Negative elements found.")

    if k > 10000:
        print(f"Projection slow (k={k}).")

    return x

def projection(x_to_proj, problem, bnds, tol=1e-8):
    """
    Project a vector based on the problem's projection type.

    Parameters:
        x_to_proj (np.ndarray): Vector to be projected.
        problem (ProblemInstance): Problem definition.
        bnds (tuple): Bounds used in projection.
        tol (float): Tolerance for projection.

    Returns:
        np.ndarray: Projected vector.

    Raises:
        ValueError: If the projection type is unsupported or required attributes are missing.
    """
    if problem.proj_type == 'Markov':
        required_attrs = ['mA', 'neighborhoods']
        for attr in required_attrs:
            if not hasattr(problem, attr):
                raise ValueError(f"ProblemInstance missing required attribute '{attr}' for Markov projection.")

        x = projection_markov(
            x_to_proj,
            bnds=bnds,
            neighborhoods=problem.neighborhoods,
            mA=problem.mA
        )

    elif problem.proj_type == 'subspace':
        if not getattr(problem, 'bProjectionReady', False):
            raise ValueError("ProblemInstance must be projection-ready for subspace projection.")
        x = dykstra(
            x_to_proj,
            tol=tol,
            A_pinv_b=problem.A_pinv_b,
            C__C_T_C_inv__C_T=problem.C__C_T_C_inv__C_T,
            bnds=bnds
        )

    else:
        raise ValueError("Unsupported projection type. Use 'Markov' or 'subspace'.")
    
    feasible(problem, x)
    return x


def y_to_x(y, A_pinv_b):
    """Converts y to x using x = A_pinv_b + y."""
    return A_pinv_b + y

def x_to_y(x, A_pinv_b):
    """Converts x to y using y = x - A_pinv_b."""
    return x - A_pinv_b

def x_to_u(x, C_pinv, A_pinv_b):
    """Converts x to u using u = C_pinv @ (x - A_pinv_b)."""
    return C_pinv @ (x - A_pinv_b)

def u_to_x(u, C, A_pinv_b):
    """Converts u to x using x = A_pinv_b + C @ u."""
    return A_pinv_b + C @ u

