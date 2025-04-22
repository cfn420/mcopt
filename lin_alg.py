# linear_algebra.py

import numpy as np
import sympy as sp
from itertools import product

def orthonormal_basis_nullspace(matrix, rank):
    """
    Compute the orthonormal basis of the nullspace of a matrix.
    
    Parameters:
      - matrix (np.ndarray): Input matrix.
      - rank (int): Rank of the matrix (number of independent constraints).
    
    Returns:
      - np.ndarray: An n-by-(n-rank) matrix whose columns form an orthonormal basis for the nullspace.
    """
    _, _, v_transpose = np.linalg.svd(matrix)
    C = np.copy(v_transpose[rank:])  # Assuming m2 is precomputed as rank(A)
    return C.T

def echelon_sympy(matrix):
    """
    Convert a matrix to row echelon form using Sympy's exact arithmetic.
    
    Parameters:
      - matrix (array-like): Input matrix to transform.
    
    Returns:
      - np.ndarray: A float64 array in row echelon form, same shape as input_matrix.
    """
    sympy_matrix = sp.Matrix(matrix.tolist())
    echelon = sympy_matrix.echelon_form()
    return np.array(echelon.tolist(), dtype=np.float64)

def echelon_form(matrix):
    """
    Perform floating-point Gaussian elimination to row echelon form.
    
    Parameters:
      - matrix (np.ndarray): Input matrix to transform.
    
    Returns:
      - np.ndarray: A copy of the input in row echelon form.
    """
    B = np.copy(matrix)
    nrows, ncols = B.shape
    j = 0
    for i in range(nrows):
        pivot_row = i
        while j < ncols and np.allclose(B[pivot_row, j], 0, rtol=1e-12, atol=1e-16):
            pivot_row += 1
            if pivot_row == nrows:
                break
        if pivot_row == nrows:
            break
        if pivot_row != i:
            B[[i, pivot_row]] = B[[pivot_row, i]]
        pivot = B[i, j]
        B[i, j:] /= pivot
        for k in range(i + 1, nrows):
            factor = B[k, j] / B[i, j]
            B[k, j:] -= factor * B[i, j:]
        j += 1
    return B

def build_row_sum_constraints(mParams, bUndirected):
    """
    Build the row-sum constraint matrix for transition parameters.
    
    Parameters:
      - mParams (np.ndarray): Parameter indicator matrix of shape (N, N).
      - bUndirected (bool): If True, treat (i,j) and (j,i) as one parameter.
    
    Returns:
      - np.ndarray: Constraint matrix of shape (N, num_params), summing outgoing transitions.
    """    
    if bUndirected:
        
        N,N = mParams.shape
        dParam = { }
        counter = 0
        for (i,j) in product(range(N), range(N)):
            if mParams[i,j] == 1:
                dParam[ (i,j) ] = counter
                counter += 1
        
        A = np.zeros((N,int(np.sum(mParams))))
        mParams_tril = np.tril(mParams.T) # lower triangular part of matrix
        np.fill_diagonal(mParams_tril,0) # set diagonal to zero to avoid 2 values in diagonal next step.
        mA = mParams + mParams_tril
        for (i,j) in product(range(N),range(N)):
            
            if mA[i,j] == 1:
                
                if i <= j:
                    A[i, dParam.get((i,j)) ] = 1 
                else:
                    A[i, dParam.get((j,i)) ] = 1 
            
        return A
    
    else:
        N,N = mParams.shape
        param_count = int(np.sum(mParams))
        param_count_row = np.sum(mParams,axis=1)
        A = np.zeros((N,param_count))
        for i in range(N):
            A[i, int(sum(param_count_row[:i])): int(sum(param_count_row[:i+1]))] = 1
        return A

def build_stationary_constraints(mA, pi_hat):
    """
    Build the stationary distribution constraint matrix A_pi.
    
    Parameters:
      - mA (np.ndarray): Adjacency indicator matrix of shape (N, N).
      - pi_hat (np.ndarray): Target stationary distribution of length N.
    
    Returns:
      - np.ndarray: Constraint matrix of shape (N, num_params) enforcing P^T pi = pi.
    """    
    N,_ = mA.shape
    dParam = { }
    counter = 0
    for (i,j) in product(range(N), range(N)):
        if mA[i,j] == 1:
            dParam[ (i,j) ] = counter
            counter += 1
    
    param_count = int(np.sum(mA))
    A_pi = np.zeros((N,param_count))
    for (i,j) in product(range(N), range(N)):
        if (i,j) in dParam.keys(): A_pi[j,dParam.get((i,j))] = pi_hat[i]
            
    return A_pi

def piv_rows(echelon_matrix):
    """
    Identify pivot rows in a row echelon matrix.
    
    Parameters:
      - echelon_matrix (np.ndarray): Matrix in row echelon form.
    
    Returns:
      - list of int: Indices of rows containing the leading non-zero entry for each pivot.
    """    
    # Identify the pivot rows and columns
    pivots = []
    pivot_columns = []
    for c, row in enumerate(echelon_matrix):
        try:
            first_nonzero_index = list(row).index(next((x for x in row if x != 0), len(row)))
        except:
            continue
            
        if first_nonzero_index not in pivot_columns:
            pivots.append(c)
            pivot_columns.append(first_nonzero_index)
            
    return pivots