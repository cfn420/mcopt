# network.py

import numpy as np
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
        
from utils import row_normalize

class MarkovChain:
    def __init__(self, mA, x=None, bUndirected=False):
        """
        Initializes a MarkovChain instance representing a Markov chain with a vector
        of transition parameters.
        
        Parameters:
            n (int): Number of nodes in the network.
            mA (np.ndarray, optional): Adjacency matrix for the network. If provided, it is used to
                compute the edge matrix. Otherwise, one is constructed from edge_indices.
            x (np.ndarray, optional): Raw vector representation for allowed transitions.
                Its length must equal the total number of allowed transitions. If None,
                it is initialized to zeros.
            bUndirected (bool, optional): If True, the network is treated as undirected (symmetrize P).
        """
        self.n = mA.shape[0]
        self.bUndirected = bUndirected

        if x is not None:
            self.x = x
        
        # Set up the adjacency matrix mA.
        self.mA = mA

        if self.bUndirected == False:
            self.edge_matrix = create_edge_matrix(self.mA)
        else:
            self.edge_matrix = create_edge_matrix(np.triu(self.mA))
        
    
    # ----------------------------
    # Conversion Methods
    # ----------------------------
    @property
    def P_matrix(self):
        """Returns the stochastic matrix obtained by directly mapping vector entries to edges."""
        if self.bUndirected == False:
            return x_to_matrix(self.x, self.n, self.edge_matrix, self.bUndirected)
        else:
            return row_normalize(x_to_matrix(self.x, self.n, self.edge_matrix, self.bUndirected))
        
    @staticmethod
    def P_to_x(P, mA, bUndirected):
        """
        Extracts the vector representation from a matrix P by reading the entries at positions
        where mA (the binary matrix) is one.
        
        Parameters:
            P (np.ndarray): Transition matrix.
            mA (np.ndarray): Binary matrix indicating allowed transitions.
            bUndirected (bool): If True, only include each undirected edge once (i <= j).
        
        Returns:
            np.ndarray: The vector of transition probabilities.
        """
        N, _ = mA.shape
        if not bUndirected:
            return np.array([P[i, j] for i, j in product(range(N), range(N)) if mA[i, j] == 1])
        else:
            return np.array([P[i, j] for i, j in product(range(N), range(N)) if mA[i, j] == 1 and i <= j])

    # ----------------------------
    # Markov Chain Properties
    # ---------------------------
    @staticmethod
    def stationary_distribution(P):
        """
        Computes the stationary distribution (π) of the Markov chain.
        
        The method uses the approach from functions.py:
        
            Let Z = P - I, with the first column replaced by ones.
            Then π is obtained from π = inv(Z)[0, :].
        
        Returns:
            np.ndarray: The stationary distribution as a 1D array.
        """
        N,N = P.shape
        Z = P - np.eye(N)
        Z[:, [0]] = np.ones((N, 1))
        pi = np.linalg.inv(Z)[0, :]
        return pi

    def kemeny_constant(self):
        """
        Computes the Kemeny constant for the Markov chain.
        
        The formula used is:
            KC = trace( inv(I - P + Pi) - Pi ) + 1
        where Pi is a matrix in which every row equals the stationary distribution.
        
        Parameters:
            use_direct (bool, optional): If True, use self.direct_P(), otherwise use softmax_P().
        
        Returns:
            float: The Kemeny constant.
        """
        P = self.P_matrix
        I = np.eye(self.n)
        pi_vec = self.stationary_distribution()
        Pi = np.tile(pi_vec, (self.n, 1))
        D = np.linalg.inv(I - P + Pi) - Pi
        return np.trace(D) + 1
    
    @staticmethod
    def M(P, pi_row):
        """
        Computes the matrix M(P) used in the objective.
        
        The formula is:
            M(P) = (I - D + Ones @ diag(diag(D))) @ inv(diag(Pi))
        where D = Z - Pi, with Z = inv(I - P + Pi) and Pi is a matrix whose rows equal the stationary distribution.
        
        Parameters:
            P (np.ndarray): A stochastic matrix.
            
        Returns:
            np.ndarray: The matrix M(P).
        """
        N,N = P.shape
        Pi = np.tile(pi_row, (N, 1))
        inv_dg_Pi = np.diag(np.diag(Pi)**-1)
        I = np.eye(N)
        Ones = np.ones((N, N))
        Z = np.linalg.inv(I - P + Pi)
        D = Z - Pi
        M_matrix = (I - D + Ones @ np.diag(np.diag(D))) @ inv_dg_Pi
        return M_matrix

    # ----------------------------
    # Static Utility Functions
    # ----------------------------
    @staticmethod
    def create_mC(N):
        """
        Creates a uniform matrix mC of size (N, N) with every entry equal to 1/(N^2).
        
        Parameters:
            N (int): Dimension of the matrix.
        
        Returns:
            np.ndarray: The uniform matrix.
        """
        return np.full((N, N), 1 / (N ** 2))
    
    @staticmethod
    def P_hat(P, mA_sample):
        P_hat = P * mA_sample
        P_hat = row_normalize(P_hat) 
        return P_hat

    # ----------------------------
    # Plotting Functions
    # ----------------------------
    def plot(self, graph_name="markov_chain", graph_title="", 
         mark_nodes=None, folder_to_save=".", save_format="png", 
         P=None, node_mask=None, show_plot=True):
        """
        Grid layout for reduced Markov chain with curved, weighted directed edges.
        """
        if P is None:
            P = self.P_matrix
        if node_mask is None:
            raise ValueError("node_mask (boolean of original nodes) is required.")
        if mark_nodes is None:
            mark_nodes = []

        N_total = len(node_mask)
        kept_nodes = np.where(node_mask)[0]
        num_nodes = len(kept_nodes)

        # Grid size (must match original)
        d = int(np.sqrt(N_total))
        if d * d != N_total:
            raise ValueError("node_mask must represent a square grid (length should be a perfect square).")

        # Map reduced matrix index (P) to original node index
        red_to_orig = kept_nodes
        orig_to_pos = {orig_idx: (orig_idx % d, (orig_idx // d)) for orig_idx in kept_nodes}

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_aspect('equal')
        ax.axis("off")

        # Plot grid placeholders (faded for missing nodes)
        for idx in range(N_total):
            x = idx % d
            y = (idx // d)
            if node_mask[idx]:
                color = '#cccccc'
            else:
                color = 'white'
            ax.add_patch(plt.Circle((x, y), 0.0001, facecolor=color, edgecolor='lightgray', linewidth=1, zorder=1))

        # Plot edges
        r = 0.2                  # Node radius
        m = 10                   # Arrowhead size
        shrink = 0.85*r  # Total shrink at receiving end

        for i_red in range(num_nodes):
            for j_red in range(num_nodes):
                weight = P[i_red, j_red]
                if weight > 1e-3:
                    i_orig = red_to_orig[i_red]
                    j_orig = red_to_orig[j_red]
                    x1, y1 = orig_to_pos[i_orig]
                    x2, y2 = orig_to_pos[j_orig]

                    if (x1, y1) == (x2, y2):
                        continue  # skip self-loops for now

                    dx, dy = x2 - x1, y2 - y1
                    distance = np.hypot(dx, dy)
                    if distance == 0:
                        continue

                    ux, uy = dx / distance, dy / distance

                    # Start at center, end just before boundary of receiver
                    start = (x1, y1)
                    end = (x2 - ux * shrink, y2 - uy * shrink)

                    path = patches.FancyArrowPatch(
                        start, end,
                        connectionstyle="arc3,rad=0.15",  # less aggressive curvature
                        arrowstyle='-|>',
                        mutation_scale=m,
                        linewidth=5 * weight,
                        color='black',
                        alpha=min(1.0, weight * 6),
                        zorder=2
                    )
                    ax.add_patch(path)

        # Plot nodes (on top)
        for i_orig in kept_nodes:
            x, y = orig_to_pos[i_orig]
            color = 'red' if i_orig in mark_nodes else '#aaaaaa'
            ax.add_patch(plt.Circle((x, y), .2, facecolor=color, edgecolor='black', linewidth=0.5, zorder=3))

        ax.set_xlim(-1, d)
        ax.set_ylim(-1, d)  

        # Save and show
        os.makedirs(folder_to_save, exist_ok=True)
        output_path = os.path.join(folder_to_save, f"{graph_name}.{save_format}")
        plt.savefig(output_path, format=save_format, dpi=300, bbox_inches='tight')


        if show_plot:
            plt.show()
        else:
            plt.close()
            print(f"Graph saved to: {output_path}")



def x_to_matrix(x, N, edge_matrix, bUndirected):
    """
    Converts the raw vector (or a given vector x) into a stochastic matrix by directly placing
    the vector values in the allowed edge positions.

    Parameters:
        x (np.ndarray): Vector of raw transition probabilities.
        N (int): Size of the output matrix (NxN).
        edge_matrix (np.ndarray): Edge index pairs indicating where to place values from x.
        bUndirected (bool): If True, the resulting matrix is symmetrized.

    Returns:
        np.ndarray: A matrix P of shape (N, N). For undirected networks the matrix is symmetrized.
    """
    P = np.zeros((N, N))
    P[edge_matrix[:, 0], edge_matrix[:, 1]] = x
    if bUndirected:
        return P + P.T
    else:
        return P
    
def remove_nodes(matrix, exclude):
    """
    Returns a submatrix obtained by excluding the rows and columns given in 'exclude'.
    (Uses the implementation from functions.py.)
    
    Parameters:
        exclude (list or np.ndarray): Indices to exclude.
        matrix (np.ndarray, optional): Matrix to subset. If None, self.direct_P() is used.
    
    Returns:
        tuple: (mSubset_matrix, mask) where mSubset_matrix is the submatrix and mask is the boolean mask.
    """
    n, _ = matrix.shape
    mask = np.full((n, n), True)
    mask[exclude] = False
    mask[:, exclude] = False
    mSubset = matrix[mask]
    mSubset_matrix = mSubset.reshape(n - len(exclude), n - len(exclude))
    return mSubset_matrix, mask

def build_node_mask(n, exclude):
    """
    Creates a boolean mask array of length n, where indices in 'exclude' are marked as False.

    Parameters:
        n (int): Total number of elements (number of nodes).
        exclude (list or np.ndarray): Indices to exclude from the mask.

    Returns:
        np.ndarray: A boolean array of shape (n,) where excluded indices are False, and all others are True.
    """
    node_mask = np.ones(n, dtype=bool)
    node_mask[exclude] = False
    return node_mask

def create_edge_matrix(mA):
    """
    Creates an edge matrix from an adjacency matrix mA.
    
    Parameters:
        mA (np.ndarray): Adjacency matrix.
    
    Returns:
        np.ndarray: An array of shape (num_edges, 2) where each row is an edge (i, j) with mA[i, j] nonzero.
    """
    indices = np.nonzero(mA)
    num_edges = indices[0].shape[0]
    E = np.zeros((num_edges, 2), dtype=int)
    E[:, 0] = indices[0]
    E[:, 1] = indices[1]
    return E

def build_neighborhoods(edge_matrix, N):
    """
    Builds a subset list for projection, grouping vector indices by source node.

    Parameters:
        edge_matrix (np.ndarray): An array of shape (num_edges, 2), where each row is an edge (i, j).
        N (int): Number of nodes in the network.

    Returns:
        list of list[int]: neighborhoods[i] contains the indices of edges starting from node i.
    """
    neighborhoods = [[] for _ in range(N)]
    for idx, (i, j) in enumerate(edge_matrix):
        neighborhoods[i].append(idx)
    return neighborhoods

