# Markov Chain SPSA Optimization

This repository implements an SPSA (Simultaneous Perturbation Stochastic Approximation) solver to optimize the transition probabilities of a Markov chain with respect to (scaled) mean first passage times as in the paper:

- Franssen, C. P., Zocca, A., & Heidergott, B. F. (2024). A First-Order Gradient Approach for the Connectivity Analysis of Markov Chains. arXiv preprint arXiv:2403.11744.

---

## 📦 Prerequisites

- Python 3.12 (tested on 3.12.2)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended for environment setup)

Install dependencies using:

```bash
conda env create -f environment.yml
conda activate mcopt
```

---

## ⚙️ Configuration

All configurable parameters are defined near the top of `main.py` under the **CONFIGURATION** block.

### 🧱 Define the network grid

```python
GRID_DIM = 9
NODES_TO_REMOVE = [11, 12, ...]
UNDIRECTED_GRAPH = False
```

### 🎯 Optimization objective

```python
PI_HAT = np.full((N,), 1/N)  # Optional stationary distribution constraint
C = np.full((N, N), 1 / N**2)  # Matrix of weights for mean first passage times.
```

### ⚙️ SPSA solver parameters

```python
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
```

### 🌩️ Failure randomness settings

```python
P_FAILURE                = 0.0    # Probability that an edge fails
CORRELATED_FAILURES      = True   # Use correlated edge failures
```

Ensure appropriate failure maps are defined in `data/failure_maps/`, which indicate the failing edges in the network (after deleting nodes from the grid, if any). For correlated failure models, a valid covariance matrix should be defined following Chen (2015), and inserted in `data/covariance_matrices/`. The current instances reflects the example used in Section VI.B in Franssen et al. (2024).

### 🧍 Transition bounds

```python
ETA = 1e-4  # Lower bound on each transition probability
```

---

## 🚀 Running the Optimizer

Run the full optimization from the terminal:

```bash
python main.py
```

This will:
- Initialize the network and Markov model
- Run SPSA optimization
- Save results under a timestamped folder inside `results/`

---

## 📊 Output & Results

Each run creates a subfolder in `results/`:

```
results/run_YYYYMMDD_HHMMSS/
├── P_opt.npy                # Optimized transition matrix
├── x_opt.npy               # Optimized decision vector
├── objective_history.png   # Plot of objective vs iteration
```

---

## 🔖 Citation

If you use this code in your work, please cite the following paper:

- Franssen, C. P., Zocca, A., & Heidergott, B. F. (2024). A First-Order Gradient Approach for the Connectivity Analysis of Markov Chains. arXiv preprint arXiv:2403.11744. 

---

## 📚 References

- Chen, M. (2015). Generating nonnegatively correlated binary random variates. The Stata Journal, 15(1), 301-308.

---

## 🛡️ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contact

Feel free to open an issue or reach out with questions at c.p.c.franssen [at] vu.nl.