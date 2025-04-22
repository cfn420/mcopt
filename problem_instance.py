# problem_instance.py

import numpy as np
from itertools import product

from lin_alg import build_row_sum_constraints, build_stationary_constraints, echelon_sympy, orthonormal_basis_nullspace, piv_rows
from network import MarkovChain, create_edge_matrix, build_neighborhoods, x_to_matrix
from utils import row_normalize

class ProblemInstance:
    def __init__(self,
                 mC,
                 mA,
                 eta,
                 p,
                 bUndirected=False,
                 pi_hat=None,
                 bUndirected_failing_edges=True,
                 bCorrelated=False,
                 mParam_failure=None,
                 mu=None,
                 rho_matrix=None,
                 all_samples=None):
        """
        Initializes the problem instance with necessary data.
        
        Parameters:
            mC (np.ndarray): Matrix of weights for mean first passage times.
            mA (np.ndarray): Adjacency matrix.
            eta (float): Lower bound edge weights.
            p (float): Probability of edge failure.
            bUndirected (bool): Whether the graph is undirected.
            pi_hat (np.ndarray, optional): Target stationary distribution constraint.
            bUndirected_failing_edges (bool): Whether failing edges are undirected.
            bCorrelated (bool): Whether failures are correlated.
            mParam_failure (np.ndarray, optional): Failure label matrix.
            mu (np.ndarray, optional): Marginal failure parameters (only for correlated case).
            rho_matrix (np.ndarray, optional): Covariance matrix (only for correlated case).
        """
        self.mC = mC
        self.mA = mA
        self.eta = eta
        self.p = p
        self.bUndirected = bUndirected
        self.N = mC.shape[0]
        self.proj_type = 'subspace'

        # Handle stationary distribution constraint
        if pi_hat is not None:
            self.pi_hat = pi_hat
            self.bPi_constraint = True
        else:
            self.bPi_constraint = False

        # Projection matrices
        self.initialize_projection_matrices()
        self.edge_matrix = create_edge_matrix(np.triu(self.mA) if self.bUndirected else self.mA)
        self.neighborhoods = build_neighborhoods(self.edge_matrix, self.N)

        # Handle randomness / failures
        if self.p > 0.:
            self.bUndirected_failing_edges = bUndirected_failing_edges
            self.bCorrelated = bCorrelated

            if mParam_failure is None:
                raise ValueError("Missing mParam_failure when p > 0.")

            self.mParam_failure = mParam_failure
            self.fail_edge_matrix = create_edge_matrix(mParam_failure)
            self.n_failing_edges = self.fail_edge_matrix.shape[0]

            # Correlated case
            if self.bCorrelated:
                if mu is None or rho_matrix is None:
                    raise ValueError("Correlated failures require both 'mu' and 'rho_matrix'.")
                self.mu = mu
                self.rho_matrix = rho_matrix

            # Samples + safe adjacency
            self.mA_safe = self.build_mA_safe()
            self.all_samples, self.sample_probabilities = self.binary_matrix_permutations() # List of all adjacency matrices and probabilities under the failure model.


    def S(self, P):
        """
        Computes a weighted sum of mean first passage times S(P) = sum( M(P) * mC ).
        
        Parameters:
            P (np.ndarray): A stochastic matrix.
            
        Returns:
            float: The scalar value of S(P).
        """
        return np.sum( MarkovChain.M(P, MarkovChain.stationary_distribution(P)) * self.mC)

    @staticmethod
    def P_hat(P, mA_sample):
        """
        Computes P̂(P, mA_sample) as the element-wise product of P and mA_sample,
        followed by row normalization.
        
        Parameters:
            P (np.ndarray): A stochastic matrix.
            mA_sample (np.ndarray): A binary sample matrix (of the same shape as P).
            
        Returns:
            np.ndarray: The redistributed stochastic matrix.
        """
        P_hat = P * mA_sample
        return row_normalize(P_hat)

    def objective(self, P, mA_sample=None):
        """
        Computes the objective value.
        
        If a sample mA_sample is provided, then the objective is evaluated as:
            S( P̂(P, mA_sample) )
        Otherwise, if sample_probabilities (a weight vector) is provided and self.all_samples is set, then the
        objective is computed as a weighted sum over samples.
        
        Parameters:
            P (np.ndarray): A stochastic matrix.
            mA_sample (np.ndarray, optional): A single sample matrix.
            
        Returns:
            float: The objective value.
        """
        if self.p == 0.0:
            return self.S(P)
        else:
            if mA_sample is not None: 
                return self.S(ProblemInstance.P_hat(P, mA_sample))        
            else : # Use presampled values (not for gradient estimation)
                return np.sum([ self.sample_probabilities[i] * self.S(ProblemInstance.P_hat(P, self.all_samples[i])) for i in range(len(self.sample_probabilities)) ])
    
    def initialize_projection_matrices(self):
        """
        Initializes and stores matrices required for affine projection onto the feasible space.

        Sets:
        - self.A, self.b: Affine constraint system.
        - self.C: Nullspace basis of A.
        - self.A_pinv_b: Affine projection base.
        - self.bProjectionReady: True when projection matrices are valid.

        """
        N = self.N
        if self.bUndirected:
            A_row = build_row_sum_constraints(np.triu(self.mA), self.bUndirected)
        else:
            A_row = build_row_sum_constraints(self.mA, self.bUndirected)
    
        if not self.bUndirected and self.bPi_constraint:
            A_pi = build_stationary_constraints(self.mA, self.pi_hat)
            A_comb = np.vstack([A_row, A_pi])
            b_comb = np.hstack([np.ones(N), self.pi_hat])[:, None]
        else:
            A_comb = A_row
            b_comb = np.ones(N)[:, None]

        A_b_comb = np.hstack([A_comb, b_comb])
        A_b_ech = echelon_sympy(A_b_comb)

        A_ech = A_b_ech[:, :-1]
        b_ech = A_b_ech[:, -1]
        pivot_rows = piv_rows(A_ech)
        A = A_ech[pivot_rows]
        b = b_ech[pivot_rows]

        self.m2, self.d = A.shape
        if np.linalg.matrix_rank(A) != len(A):
            raise Warning("Echelon operations incomplete or matrix is rank-deficient.")

        A_pinv = A.T @ np.linalg.inv(A @ A.T)
        A_pinv_b = A_pinv @ b

        C = orthonormal_basis_nullspace(A, self.m2)
        C__C_T_C_inv__C_T = C @ C.T

        # Attach to the instance
        self.A = A
        self.b = b
        self.C = C
        self.A_pinv_b = A_pinv_b
        self.C__C_T_C_inv__C_T = C__C_T_C_inv__C_T
        self.bProjectionReady = True

    def sample_correlated_adj(self):
        """
        Generates a sample failure matrix using a correlated multivariate normal model.

        Returns:
            np.ndarray: Sampled adjacency matrix with correlated edge failures applied.
        """
        X = np.random.multivariate_normal(self.mu, self.rho_matrix, 1)
        Y = np.zeros(X.shape)
        Y[X > 0] = 1
        Y = np.abs(Y-1)
        
        mA_sample = x_to_matrix( Y, self.N, self.fail_edge_matrix, self.bUndirected ) + self.mA_safe
        return mA_sample

    def sample_mA(self):
        """
        Draws a sample failure matrix mA_sample, based on either a correlated
        or uncorrelated model, using the instance's parameters.

        Returns:
            np.ndarray: A sampled mA matrix.
        """
        if self.p==0.0:
            return self.mA

        if self.bCorrelated:
            return self.sample_correlated_adj()

        sample_vector = np.random.choice(
            [0, 1],
            p=[self.p, 1 - self.p],
            size=self.n_failing_edges
        )
        mA_sample = x_to_matrix(
            sample_vector,
            self.N,
            self.fail_edge_matrix,
            bUndirected=self.bUndirected
        )
        return mA_sample + self.mA_safe

    def build_mA_safe(self):
        """
        Builds a matrix representing the structure with all possible failing edges removed.

        Returns:
            np.ndarray: The 'safe' adjacency matrix (unaffected by failure sampling).

        Raises:
            Warning: If matrix is claimed undirected but not symmetric.

        """
        if self.bUndirected_failing_edges:
            if np.sum( np.tril(self.mParam_failure) ) > 0:
                raise Warning('mParam_failure does not appear undirected.')
            return self.mA - (self.mParam_failure + self.mParam_failure.T)
        else:
            return self.mA - (self.mParam_failure)

    def binary_matrix_permutations(self):
        """
        Generates all 2ⁿ binary samples over the failing edges and computes their probabilities.

        Returns:
            tuple:
                - list of np.ndarray: Sampled adjacency matrices with failed edges.
                - np.ndarray: Associated probabilities (same order).
        """
        # Find the coordinates (row, col) of 1 values in the original matrix
        ones_coords = np.argwhere(self.mParam_failure > 0)
        
        def generate_binary_tuples(n):
            binary_set = product([0, 1], repeat=n)
            return binary_set
        
        # Generate all possible permutations of the 1 values
        permuted_matrices = [] # list of sampled matrices
        mat_probabilities = [] # probabilities of each permutation
        for perm in generate_binary_tuples(len(ones_coords)):
            
            permuted_matrix = np.zeros_like(self.mParam_failure)
            vProb = []
            for i, b in enumerate(perm):
                
                # Compute permuted matrix
                r, c = ones_coords[i][0], ones_coords[i][1]
                permuted_matrix[r,c] = b
                
                # Compute probability
                if b == 1:
                    vProb.append(1 - self.p)
                elif b==0:
                    vProb.append(self.p)
                else:
                    raise Warning('')
            
            if self.bUndirected_failing_edges: 
                permuted_matrix += permuted_matrix.T 
            
            sampled_mA  = self.mA_safe + permuted_matrix
            permuted_matrices.append(sampled_mA)
            mat_probabilities.append( np.prod(np.array(vProb)) )
        
        if self.bUndirected_failing_edges: 
            mat_probabilities = mat_probabilities / np.sum(mat_probabilities)
        
        return permuted_matrices, mat_probabilities