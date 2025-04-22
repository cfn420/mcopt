# utils.py

import os
import pandas as pd
import numpy as np
import pickle

def row_normalize(M):
    """
    Normalizes the rows of matrix M so that each row sums to 1.
    
    Parameters:
        M (np.ndarray): Input matrix.
    
    Returns:
        np.ndarray: A matrix where each row is normalized to sum to 1.
    """
    # Compute the sum of each row; keep dimensions for broadcasting
    row_sums = np.sum(M, axis=1, keepdims=True)
    # Avoid division by zero: if a row sum is zero, set it to one (row remains unchanged)
    row_sums[row_sums == 0] = 1
    return M / row_sums

def load_excel_matrix(path):
    """Load an Excel file and return a NumPy matrix."""
    df = pd.read_excel(path, header=None).fillna(0)
    return df.to_numpy()

def save_var_pickle(var, directory, file_name):
    """Save a variable to a pickle file in a given directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name + '.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(var, f)

def read_pickle(file_path):
    """Read a variable from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def folder(directory, folder_name):
    """Create a folder if it does not exist."""
    path = os.path.join(directory, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_results(x_opt, p_opt, output_dir):
    """Save optimized variables to .npy files in `output_dir`."""
    np.save(os.path.join(output_dir, "x_opt.npy"), x_opt)
    np.save(os.path.join(output_dir, "P_opt.npy"), p_opt)
