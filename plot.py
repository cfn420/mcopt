# plot.py

import os
from datetime import datetime
import matplotlib.pyplot as plt

def create_output_dir(base_dir="results"):
    """
    Create a timestamped output directory under `base_dir`.

    Returns:
        The path to the new directory.
    """
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{time_stamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_objective_history(obj_history, output_dir, filename="objective_history.png"):
    """
    Plot and save the SPSA objective history.

    Returns:
        The full path to the saved figure.
    """
    plt.figure()
    plt.plot(obj_history)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("SPSA Objective History")
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    return path

def plot_markov_chain(mc, graph_name="example", graph_title="My Markov Chain", node_mask=None):
    """
    Invoke the MarkovChain plot method and optionally save or move outputs.
    """
    mc.plot(graph_name=graph_name, graph_title=graph_title, node_mask=node_mask)