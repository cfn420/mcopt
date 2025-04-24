#!/usr/bin/env python3
"""
main.py

This file initializes the optimization problem, constructs the network,
and runs the SPSA solver to minimize the objective.
"""

import logging
import time
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.stats import norm
import copy
from types import SimpleNamespace

from optimization.spsa import solve_spsa
from optimization.feasibility import feasible
from network import remove_nodes
from initialization import sample_best
from problem_instance import ProblemInstance
from plot import create_output_dir, plot_objective_history
from utils import load_excel_matrix, read_pickle, save_results


# ─── CONFIGURATION ───────────────────────────────────────────

# Project directories
PROJECT_ROOT    = Path(__file__).parent.resolve()
DATA_DIR        = PROJECT_ROOT / "data"
FAILURE_DIR     = DATA_DIR    / "failure_maps"
RESULTS_DIR     = PROJECT_ROOT / "results"
COV_DIR         = DATA_DIR    / "covariance_matrices"

# Excel files
FAIL_LABEL_XLSX     = FAILURE_DIR / "failure_labeling.xlsx"
FAIL_LABEL_COR_XLSX = FAILURE_DIR / "failure_labeling_cor.xlsx"
COV_MATRIX_XLSX     = COV_DIR / "cov_matrix_p_0.5_5.pickle"

# Network settings
GRID_DIM          = 9
NODES_TO_REMOVE   = [11, 12, 20, 21, 22, 23, 29, 30, 31, 32, 58, 59, 68]
UNDIRECTED_GRAPH  = False

# Problem data
N      = GRID_DIM * GRID_DIM - len(NODES_TO_REMOVE)  # Number of nodes in the network
PI_HAT = np.full(shape=(N), fill_value=1/N)  # None # Stationary distribution constraint (None if not used)
C      = np.full((N, N), 1 / N**2)

# SPSA solver parameters
SPSA_CONFIG = SimpleNamespace(
    max_iter=5_000_000, # Number of iterations
    a=0.1, # Step size for the gain
    e=1e-8, # Perturbation size
    r_epsilon=0.602, # Exponent for the gain
    r_nu=0.200,     # Exponent for the perturbation
    a_eps=1_000_000, # Step size denumqinator add-on
    obj_interval=1000, # Interval for objective function evaluation
    nsamples=1, # Number of single-run SPSA estimators
    omega=1e-3, # Convergence threshold
    target=None, # Target value for the objective function (None if not used)
)

# Failure‐randomness settings
P_FAILURE                = 0.0    # Probability that an edge fails
CORRELATED_FAILURES      = True   # Use correlated edge failures

# Lower bound for transition probabilities
ETA = 1e-4

# ─── END CONFIGURATION ────────────────────────────────────────

def main():

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    np.random.seed(12345)
    
    # Build the network graph
    G_nx    = nx.grid_2d_graph(GRID_DIM, GRID_DIM)
    mA_full = nx.to_numpy_array(G_nx)
    mA, _   = remove_nodes(mA_full, NODES_TO_REMOVE)

    # Failure‐randomness configuration (only if P_FAILURE > 0)
    randomness_kwargs = {}
    if P_FAILURE > 0.0:

        # 1) pick the right Excel file
        excel_path = (
            FAIL_LABEL_COR_XLSX if CORRELATED_FAILURES
            else FAIL_LABEL_XLSX
        )

        # 2) load the label matrix of failures
        mp_label = load_excel_matrix(str(excel_path))

        # 3) infer how many distinct failure‐labels
        num_failure_labels = int(np.nanmax(mp_label))

        # 4) build failure distribution
        vQ = np.full(num_failure_labels, P_FAILURE) if CORRELATED_FAILURES else None
        mu  = norm.ppf(vQ) if CORRELATED_FAILURES else None
        rho_matrix = read_pickle(COV_MATRIX_XLSX) if CORRELATED_FAILURES else None

        # 5) which edges can fail?
        mParam_failure   = (mp_label > 0).astype(int)

        randomness_kwargs = dict(
            bUndirected_failing_edges=True,
            bCorrelated=CORRELATED_FAILURES,
            mu=mu,
            rho_matrix=rho_matrix,
            mParam_failure=mParam_failure,
        )

    # Instantiate the problem
    problem = ProblemInstance(
        mC=C,
        mA=mA,
        eta=ETA,
        p=P_FAILURE,
        bUndirected=UNDIRECTED_GRAPH,
        pi_hat=PI_HAT,
        **randomness_kwargs,
    )

    # Initialize the Markov‐chain object
    logger.info("Sampling intials…")
    mc = sample_best(problem)

    # Run SPSA
    logger.info("Starting SPSA optimization…")
    t0 = time.time()
    x_hist, obj_hist = solve_spsa(
        problem, mc, config=SPSA_CONFIG
    )
    logger.info(f"Optimization finished in {time.time() - t0:.1f}s")

    # Reconstruct optimal chain
    k      = x_hist.shape[0] # Number of iterations used
    mc_opt = copy.deepcopy(mc)
    mc_opt.x = x_hist[int(0.5 * k) :, :].mean(axis=0) # Polyak-Ruppert averaging
    feasible(problem, mc_opt.x) # Check feasibility
    P_opt = mc_opt.P_matrix     # Optimal transition matrix

    # Create results folder
    out_dir = create_output_dir(base_dir=str(RESULTS_DIR))

    # Produce plots
    plot_objective_history(obj_hist, out_dir)

    # Save arrays
    save_results(x_hist, P_opt, out_dir)

    
if __name__ == "__main__":
    main()
