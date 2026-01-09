"""
Brownian Motion Simulation Module

This module provides functionality to generate and visualize Brownian motion paths.
Demonstrates the Central Limit Theorem by showing convergence from uniform to normal distribution.
"""
print("Loading browninan.py module...")
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from scipy import stats

print("browninan.py module loaded successfully.")
print("")

class BrownianMotion:
    """
    A class to simulate Brownian motion in n-dimensional space.
    
    Attributes:
        dim (int): Dimensionality of the Brownian motion (default: 2)
        steps (int): Number of time steps in the simulation (default: 1000)
        path (np.ndarray): Array containing the generated path coordinates
        step_distribution (str): Distribution type for steps ('uniform' or 'gaussian')
        n_sum (int): Number of uniform random variables to sum (for CLT demonstration)
    """
    
    def __init__(self, dim: int = 2, steps: int = 1000, seed: int = None, 
                 step_distribution: str = 'gaussian', n_sum: int = 1):
        """
        Initialize the Brownian motion simulator.
        
        Args:
            dim: Number of dimensions for the motion
            steps: Number of time steps to simulate
            seed: Random seed for reproducibility (optional)
            step_distribution: 'uniform' or 'gaussian' for step generation
            n_sum: Number of uniform variables to sum (demonstrates CLT)
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.dim = dim
        self.steps = steps
        self.step_distribution = step_distribution
        self.n_sum = n_sum
        self.path, self.increments = self._generate_path()
    
    def _generate_path(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a Brownian motion path using specified distribution.
        
        Returns:
            Tuple of (path, increments) arrays
        """
        if self.step_distribution == 'uniform':
            # Generate steps from uniform distribution [-0.5, 0.5]
            if self.n_sum == 1:
                increments = np.random.uniform(-0.5, 0.5, size=(self.steps, self.dim))
            else:
                # Sum multiple uniform random variables (CLT demonstration)
                increments = np.zeros((self.steps, self.dim))
                for _ in range(self.n_sum):
                    increments += np.random.uniform(-0.5, 0.5, size=(self.steps, self.dim))
                # Normalize to maintain similar scale
                increments /= np.sqrt(self.n_sum / 12)  # Variance of uniform[-0.5,0.5] is 1/12
        else:
            # Generate random steps from normal distribution
            increments = np.random.normal(0, 1, size=(self.steps, self.dim))
        
        # Compute cumulative sum to get positions
        path = np.cumsum(increments, axis=0)
        
        # Start from origin
        path = np.vstack([np.zeros(self.dim), path])
        
        return path, increments
    
    def plot(self, ax=None, **kwargs) -> None:
        """
        Plot the Brownian motion path.
        
        Args:
            ax: Matplotlib axis object (optional)
            **kwargs: Additional plotting arguments
        
        Raises:
            NotImplementedError: If dimensionality is not 2 or 3
        """
        if self.dim == 2:
            self._plot_2d(ax, **kwargs)
        elif self.dim == 3:
            self._plot_3d(ax, **kwargs)
        else:
            raise NotImplementedError(
                f"Plotting is only implemented for 2D and 3D Brownian motion. Got {self.dim}D."
            )
    
    def _plot_2d(self, ax=None, **kwargs) -> None:
        """Plot 2D Brownian motion."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(self.path[:, 0], self.path[:, 1], **kwargs)
        ax.scatter(0, 0, c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(self.path[-1, 0], self.path[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
        
        title = f"2D Brownian Motion ({self.step_distribution}"
        if self.step_distribution == 'uniform' and self.n_sum > 1:
            title += f", n_sum={self.n_sum}"
        title += ")"
        
        ax.set_title(title)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if ax is None:
            plt.show()
    
    def _plot_3d(self, ax=None, **kwargs) -> None:
        """Plot 3D Brownian motion."""
        from mpl_toolkits.mplot3d import Axes3D
        
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(self.path[:, 0], self.path[:, 1], self.path[:, 2], **kwargs)
        ax.scatter(0, 0, 0, c='green', s=100, marker='o', label='Start')
        ax.scatter(self.path[-1, 0], self.path[-1, 1], self.path[-1, 2], 
                   c='red', s=100, marker='x', label='End')
        ax.set_title("3D Brownian Motion")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.legend()
        
        if ax is None:
            plt.show()


def plot_multiple_paths(dim: int = 2, steps: int = 1000, n_paths: int = 10, 
                        figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot multiple Brownian motion paths on the same figure.
    
    Args:
        dim: Dimensionality of the motion
        steps: Number of time steps per path
        n_paths: Number of paths to generate and plot
        figsize: Figure size as (width, height)
    
    Raises:
        NotImplementedError: If dimensionality is not 2 or 3
    """
    if dim not in [2, 3]:
        raise NotImplementedError(
            f"Plotting is only implemented for 2D and 3D Brownian motion. Got {dim}D."
        )
    
    if dim == 2:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    
    for i in range(n_paths):
        bm = BrownianMotion(dim=dim, steps=steps)
        bm.plot(ax=ax, linewidth=1, alpha=0.7, label=f"Path {i + 1}")
    
    ax.set_title(f"{n_paths} Examples of {dim}D Brownian Motion")
    ax.legend(loc="best", fontsize="small", ncol=2 if n_paths > 10 else 1)
    plt.tight_layout()
    plt.show()


def visualize_clt_convergence(dim: int = 2, steps: int = 1000, 
                              n_sum_values: List[int] = [1, 5, 10, 30],
                              n_samples: int = 10000) -> None:
    """
    Visualize the Central Limit Theorem by showing how uniform steps converge to Gaussian.
    
    Args:
        dim: Dimensionality of the motion
        steps: Number of time steps per path
        n_sum_values: List of n_sum values to demonstrate convergence
        n_samples: Number of samples for histogram
    """
    fig, axes = plt.subplots(2, len(n_sum_values), figsize=(16, 8))
    
    for idx, n_sum in enumerate(n_sum_values):
        # Generate samples
        if n_sum == 1:
            samples = np.random.uniform(-0.5, 0.5, size=n_samples)
        else:
            samples = np.zeros(n_samples)
            for _ in range(n_sum):
                samples += np.random.uniform(-0.5, 0.5, size=n_samples)
            samples /= np.sqrt(n_sum / 12)
        
        # Plot histogram
        ax_hist = axes[0, idx]
        counts, bins, _ = ax_hist.hist(samples, bins=50, density=True, 
                                        alpha=0.7, color='blue', edgecolor='black')
        
        # Overlay Gaussian
        mu, sigma = samples.mean(), samples.std()
        x = np.linspace(samples.min(), samples.max(), 100)
        ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                     label=f'N({mu:.2f}, {sigma:.2f}²)')
        
        ax_hist.set_title(f'n_sum = {n_sum}')
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Plot Q-Q plot
        ax_qq = axes[1, idx]
        stats.probplot(samples, dist="norm", plot=ax_qq)
        ax_qq.set_title(f'Q-Q Plot (n_sum = {n_sum})')
        ax_qq.grid(True, alpha=0.3)
    
    plt.suptitle('Central Limit Theorem: Convergence from Uniform to Gaussian', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def compare_distributions(dim: int = 2, steps: int = 1000, 
                         n_sum_values: List[int] = [1, 5, 30]) -> None:
    """
    Compare Brownian motion paths generated with different distributions.
    
    Args:
        dim: Dimensionality of the motion
        steps: Number of time steps per path
        n_sum_values: List of n_sum values to compare
    """
    n_plots = len(n_sum_values) + 1  # +1 for Gaussian
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot uniform distributions with different n_sum
    for idx, n_sum in enumerate(n_sum_values):
        bm = BrownianMotion(dim=dim, steps=steps, step_distribution='uniform', n_sum=n_sum)
        ax = axes[idx]
        ax.plot(bm.path[:, 0], bm.path[:, 1], linewidth=1, alpha=0.8)
        ax.scatter(0, 0, c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(bm.path[-1, 0], bm.path[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
        ax.set_title(f'Uniform (n_sum={n_sum})')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # Plot Gaussian distribution
    bm_gauss = BrownianMotion(dim=dim, steps=steps, step_distribution='gaussian')
    ax = axes[-1]
    ax.plot(bm_gauss.path[:, 0], bm_gauss.path[:, 1], linewidth=1, alpha=0.8)
    ax.scatter(0, 0, c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(bm_gauss.path[-1, 0], bm_gauss.path[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    ax.set_title('Gaussian')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.suptitle('Brownian Motion: Uniform → Gaussian (CLT)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_displacement_distribution(steps: int = 1000, n_sum_values: List[int] = [1, 5, 10, 30],
                                       n_paths: int = 1000) -> None:
    """
    Visualize the distribution of final displacements for different n_sum values.
    
    Args:
        steps: Number of time steps per path
        n_sum_values: List of n_sum values to compare
        n_paths: Number of paths to simulate for statistics
    """
    fig, axes = plt.subplots(1, len(n_sum_values), figsize=(16, 4))
    
    if len(n_sum_values) == 1:
        axes = [axes]
    
    for idx, n_sum in enumerate(n_sum_values):
        displacements = []
        for _ in range(n_paths):
            bm = BrownianMotion(dim=2, steps=steps, step_distribution='uniform', n_sum=n_sum)
            displacement = np.sqrt(bm.path[-1, 0]**2 + bm.path[-1, 1]**2)
            displacements.append(displacement)
        
        ax = axes[idx]
        ax.hist(displacements, bins=40, density=True, alpha=0.7, color='blue', edgecolor='black')
        
        # Overlay theoretical Rayleigh distribution for 2D
        displacements = np.array(displacements)
        sigma = displacements.std() / np.sqrt(2 - np.pi/2)
        x = np.linspace(0, displacements.max(), 100)
        rayleigh = stats.rayleigh.pdf(x, scale=sigma)
        ax.plot(x, rayleigh, 'r-', linewidth=2, label=f'Rayleigh(σ={sigma:.2f})')
        
        ax.set_title(f'Final Displacement (n_sum={n_sum})')
        ax.set_xlabel('Distance from Origin')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Final Displacements: Convergence to Rayleigh', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example 1: Visualize CLT convergence
    print("Visualizing Central Limit Theorem convergence...")
    visualize_clt_convergence(steps=1000, n_sum_values=[1, 5, 10, 30])
    
    # Example 2: Compare paths with different distributions
    print("Comparing Brownian motion paths...")
    compare_distributions(dim=2, steps=1000, n_sum_values=[1, 5, 30])
    
    # Example 3: Visualize displacement distributions
    print("Visualizing displacement distributions...")
    visualize_displacement_distribution(steps=1000, n_sum_values=[1, 5, 10, 30], n_paths=1000)
    
    # Example 4: Multiple paths comparison
    print("Generating multiple 2D Brownian motion paths...")
    plot_multiple_paths(dim=2, steps=1000, n_paths=5)
