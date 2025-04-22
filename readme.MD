# Markov Chain SPSA Optimization

This repository implements a stochastic approximation (SPSA) solver to optimize transition probabilities of a Markov chain under possible edge failures.

---

## Prerequisites

- Python 3.7+

You can install dependencies via:

```bash
conda env create -f environment.yml
conda activate mcopt
```


## Configuration

All user‑tweakable settings live at the top of `main.py` under the **CONFIGURATION** block:

```python
# Project directories
GRID_DIM          = 9                   # Grid graph dimension
NODES_TO_REMOVE   = [11,12,20,...]      # Nodes to exclude
UNDIRECTED_GRAPH  = True                # Use undirected graph?

# Failure randomness
P_FAILURE                = 0.5         # Probability an edge fails
UNDIRECTED_FAILING_EDGES = True        # Are failures undirected?
CORRELATED_FAILURES      = False       # Use correlated failures?

# SPSA solver parameters
SPSA_CONFIG = SimpleNamespace(
    max_iter=5_000_000,
    a=0.1,
    e=1e-8,
    r_epsilon=0.602,
    r_nu=0.200,
    a_eps=1_000_000,
    obj_interval=1000,
    nsamples=1,
    omega=1e-3,
    target=None,
)

# Data file paths
DATA_DIR        = Path(__file__).parent / "data"
FAIL_LABEL_XLSX = DATA_DIR / "failure_maps" / "failure_labeling.xlsx"
```

Adjust any of these values and rerun `main.py` to apply new settings.


## Usage

Run the optimization end‑to‑end with:

```bash
python main.py
```

## Examples

After running, open the generated `results/run_<timestamp>/objective_history.png` to see convergence of the objective function, and optimal Markov chain `P_opt.npy`


## License & Contact

This code is released under the MIT License.  
Questions or feedback? Open an issue or contact the maintainer.

