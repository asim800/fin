import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass

@dataclass
class MJDParams:
    mu: float        # Drift
    sigma: float     # Diffusion Volatility
    lam: float       # Jump Intensity (jumps per year)
    mu_j: float      # Mean of Log-Jump Size
    delta_j: float   # Std of Log-Jump Size

class MertonJumpDiffusion:
    def __init__(self, params: MJDParams = None):
        self.params = params

    def simulate_path(self, S0, T, dt=1/252, seed=None):
        """
        Simulates a single asset path using MJD.
        """
        if seed: np.random.seed(seed)
        
        n_steps = int(T / dt)
        time = np.linspace(0, T, n_steps)
        prices = np.zeros(n_steps)
        prices[0] = S0
        
        # Unpack parameters
        mu, sigma = self.params.mu, self.params.sigma
        lam, mu_j, delta_j = self.params.lam, self.params.mu_j, self.params.delta_j
        
        # 1. Diffusion component (Brownian Motion)
        Z = np.random.normal(0, 1, size=n_steps-1)
        diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        
        # 2. Jump component (Compound Poisson)
        # Poisson number of jumps in each step (usually 0 or 1 for small dt)
        n_jumps = np.random.poisson(lam * dt, size=n_steps-1)
        
        # Magnitude of jumps
        jump_log_returns = np.zeros_like(diffusion)
        for i, n in enumerate(n_jumps):
            if n > 0:
                # Sum of n independent log-normal jumps
                jump_sum = np.sum(np.random.normal(mu_j, delta_j, size=n))
                jump_log_returns[i] = jump_sum
        
        # 3. Combine
        log_returns = diffusion + jump_log_returns
        prices[1:] = S0 * np.exp(np.cumsum(log_returns))
        
        return time, prices

    def estimate_from_returns(self, returns_series, dt=1/252, threshold_sigma=3.0):
        """
        Estimates parameters using the Iterative Thresholding method.
        returns_series: pd.Series of log returns
        threshold_sigma: Number of std devs to classify a return as a jump
        """
        # 1. Initial robust volatility estimate (Bipower variation or MAD)
        # Using simple MAD (Median Absolute Deviation) for robustness against outliers
        median = np.median(returns_series)
        mad = np.median(np.abs(returns_series - median))
        robust_sigma = mad * 1.4826 / np.sqrt(dt)  # 1.4826 scales MAD to Sigma for Normal
        
        # 2. Identify Jumps
        # A return is a jump if abs(return) > threshold * robust_sigma * sqrt(dt)
        jump_threshold = threshold_sigma * robust_sigma * np.sqrt(dt)
        
        is_jump = np.abs(returns_series) > jump_threshold
        jumps = returns_series[is_jump]
        diffusion_returns = returns_series[~is_jump]
        
        # 3. Estimate Diffusion Parameters (from non-jump data)
        mu_hat = diffusion_returns.mean() / dt + 0.5 * robust_sigma**2 
        sigma_hat = diffusion_returns.std() / np.sqrt(dt)
        
        # 4. Estimate Jump Parameters (from jump data)
        n_jumps = len(jumps)
        total_time = len(returns_series) * dt
        
        if n_jumps > 0:
            lam_hat = n_jumps / total_time
            mu_j_hat = jumps.mean()
            delta_j_hat = jumps.std()
        else:
            # Fallback if no jumps detected
            lam_hat = 0.0
            mu_j_hat = 0.0
            delta_j_hat = 0.0
            
        estimated_params = MJDParams(
            mu=mu_hat, 
            sigma=sigma_hat, 
            lam=lam_hat, 
            mu_j=mu_j_hat, 
            delta_j=delta_j_hat
        )
        self.params = estimated_params
        return estimated_params

# --- Example Usage ---
if __name__ == "__main__":
    # Define Ground Truth
    true_params = MJDParams(mu=0.1, sigma=0.2, lam=2.0, mu_j=-0.05, delta_j=0.1)
    
    # Simulate a long history (e.g., 10 years) to recover params
    model = MertonJumpDiffusion(true_params)
    t, S = model.simulate_path(S0=100, T=10, dt=1/252, seed=42)
    
    # Calculate Log Returns
    log_returns = np.diff(np.log(S))
    
    # Estimate
    est_model = MertonJumpDiffusion()
    estimated = est_model.estimate_from_returns(log_returns, dt=1/252)
    
    print("True Parameters:", true_params)
    print("Estimated:", estimated)
