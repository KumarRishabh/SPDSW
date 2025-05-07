
import torch
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from scipy.stats import wishart
from scipy.stats import multivariate_normal
import geoopt


class Betaprime:
    """
    This class implements the BetaPrime kernel, over the SPD manifold.
    
    The BetaPrime kernel is defined as:

        k(x, y) = (x^T y)^2 / (||x||^2 ||y||^2)
    where x and y are SPD matrices.
    """

    def __init__(self, d, n_proj, device="cpu", dtype=torch.float64):
        """
        Initialize the BetaPrime kernel.

        Parameters:
        - d: Dimension of the SPD matrices.
        - n_proj: Number of projections for the kernel.
        - device: Device to use for computations (default: "cpu").
        - dtype: Data type for computations (default: torch.float64).
        """
        self.d = d
        self.n_proj = n_proj
        self.device = device
        self.dtype = dtype
        self.eye = torch.eye(d, dtype=dtype, device=device)

    def kernel(self, x, y):
        """
        Compute the BetaPrime kernel between two SPD matrices.

        Parameters:
        - x: First SPD matrix (shape: [d, d]).
        - y: Second SPD matrix (shape: [d, d]).

        Returns:
        - k: The BetaPrime kernel value.
        """
        x = x.view(self.d, self.d)
        y = y.view(self.d, self.d)

        k = torch.bmm(x.unsqueeze(0), y.unsqueeze(0).transpose(1, 2))
        k = torch.bmm(k, x.unsqueeze(0).transpose(1, 2))
        k = k / (torch.norm(x) ** 2 * torch.norm(y) ** 2)

        return k
    
    def compute(self, x0, x1):
        """
        Compute the BetaPrime kernel between two batches of SPD matrices.

        Parameters:
        - x0: First batch of SPD matrices (shape: [n_samples, d, d]).
        - x1: Second batch of SPD matrices (shape: [n_samples, d, d]).

        Returns:
        - k: The BetaPrime kernel values (shape: [n_samples]).
        """
        k = torch.zeros(x0.shape[0], dtype=self.dtype, device=self.device)

        for i in range(x0.shape[0]):
            k[i] = self.kernel(x0[i], x1[i])

        return k
    
    def __call__(self, x0, x1):
        """
        Call the BetaPrime kernel.

        Parameters:
        - x0: First batch of SPD matrices (shape: [n_samples, d, d]).
        - x1: Second batch of SPD matrices (shape: [n_samples, d, d]).

        Returns:
        - k: The BetaPrime kernel values (shape: [n_samples]).
        """
        return self.compute(x0, x1)
    
# Example usage
if __name__ == "__main__":
    d = 3
    n_proj = 10
    device = "cpu"
    dtype = torch.float64

    # Create random SPD matrices
    x0 = torch.rand(5, d, d, dtype=dtype)
    x1 = torch.rand(5, d, d, dtype=dtype)

    # Initialize the BetaPrime kernel
    beta_prime_kernel = Betaprime(d, n_proj, device=device, dtype=dtype)

    # Compute the kernel values
    k_values = beta_prime_kernel(x0, x1)
    print(k_values)
