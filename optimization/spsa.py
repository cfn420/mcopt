import logging
import numpy as np
import time
from tqdm import trange
import keyboard
import copy

from optimization.projections import projection
from optimization.feasibility import feasible

logger = logging.getLogger(__name__)

def g_spsa(problem, x, L, k, mc_plus, mc_min, config=None):
    """
    Estimate the gradient using the Simultaneous Perturbation Stochastic Approximation (SPSA) method.

    Parameters:
        problem (ProblemInstance): The optimization problem instance.
        x (np.ndarray): Current decision vector.
        L (int): Number of independent gradient estimates to average.
        k (int): Current iteration (used for step size scheduling).
        mc_plus (MarkovChain): A copy of the Markov chain for forward perturbation.
        mc_min (MarkovChain): A copy of the Markov chain for backward perturbation.
        config: SPSA configuration namespace with attributes e, r_nu.

    Returns:
        np.ndarray: Averaged gradient estimate (shape: d).
    """
    m2 = problem.m2
    d = problem.d
    C = problem.C

    vG = np.zeros((L, d))
    mObj = np.zeros((L, 2))
    mDelta = np.random.choice([-1, 1], size=(L, d - m2))

    for l in range(L):
        step_size = nu_fn(k, config.e, config.r_nu)
        direction = C @ mDelta[l]

        mc_plus.x = x + step_size * direction
        mc_min.x = x - step_size * direction

        mA_sample = problem.sample_mA()
        mObj[l, 0] = problem.objective(mc_plus.P_matrix, mA_sample=mA_sample)
        mObj[l, 1] = problem.objective(mc_min.P_matrix, mA_sample=mA_sample)

        vG[l] = C @ ((mObj[l, 0] - mObj[l, 1]) / (2 * step_size * mDelta[l]))

    return np.mean(vG, axis=0)


def solve_spsa(problem, mc, config=None):
    """
    Run the SPSA optimization algorithm.

    Parameters:
        problem (ProblemInstance): The problem instance to optimize.
        mc (MarkovChain): Initial Markov chain object.
        config (SimpleNamespace): Configuration with SPSA parameters.

    Returns:
        tuple:
            - vX (np.ndarray): Array of decision vectors over iterations.
            - vObj (np.ndarray): Array of objective values over iterations.
    """
    max_iter = config.max_iter
    vX = np.zeros((max_iter, len(mc.x)))
    vX[0] = mc.x

    mc_plus = copy.deepcopy(mc)
    mc_min = copy.deepcopy(mc)

    vObj = np.zeros(max_iter)
    start_obj_interval = time.time()

    for k in trange(max_iter - 1):
        # Objective evaluation
        if k % config.obj_interval == 0:
            logger.info(f"\n[Iter {k}] Time since last obj eval: {time.time() - start_obj_interval:.2f}s")
            mc.x = vX[k]
            feasible(problem, mc.x)

            if k == 0:
                vObj[k] = problem.objective(mc.P_matrix)
            else:
                # Polyak-Ruppert averaging
                mc.x = np.mean(vX[int(0.5 * k):k, :], axis=0)
                feasible(problem, mc.x)
                vObj[k] = problem.objective(mc.P_matrix)

                # Check convergence
                if config.target is not None and vObj[k] < config.target:
                    logger.info(f"Target reached at iteration {k}.")
                    break
                elif config.target is None and abs(vObj[k] - vObj[k - config.obj_interval]) < config.omega:
                    logger.info(f"Converged (Δobj < {config.omega}) at iteration {k}.")
                    break

            logger.info(f"Objective value: {vObj[k]:.6f} / Target: {config.target}")
            start_obj_interval = time.time()
        else:
            vObj[k] = vObj[k - 1]

        # Gradient update
        vG = g_spsa(problem, vX[k], config.nsamples, k, mc_plus, mc_min, config=config)
        step_size = epsilon_fn(k, config.a, config.a_eps, config.r_epsilon)
        vX[k + 1] = vX[k] - step_size * vG

        # Projection
        if np.any(vX[k + 1] < problem.eta):
            vX[k + 1] = projection(vX[k + 1], problem, bnds=(problem.eta, 1 - problem.eta))

        # Exit if ESC pressed
        if keyboard.is_pressed('esc'):
            logger.warning("Optimization manually interrupted at iteration %d.", k)
            break

    return vX[:k + 1], vObj[:k + 1]


def epsilon_fn(k, a=1, a_eps=0, r_epsilon=0.602):
    """
    Step size schedule for the SPSA algorithm.

    Parameters:
        k (int): Current iteration.
        a (float): Initial step size coefficient.
        a_eps (float): Stability constant.
        r_epsilon (float): Decay exponent.

    Returns:
        float: Step size at iteration k.
    """
    return a / (a_eps + k + 1)**r_epsilon


def nu_fn(k, e=1, r_nu=0.101):
    """
    Perturbation size schedule for SPSA.

    Parameters:
        k (int): Current iteration.
        e (float): Initial perturbation size.
        r_nu (float): Decay exponent.

    Returns:
        float: Perturbation size at iteration k.
    """
    return e / (k + 1)**r_nu
