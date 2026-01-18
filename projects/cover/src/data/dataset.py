import torch
from torch.utils.data import Dataset
import numpy as np
from ..training.noise_schedules import cosine_beta_schedule, simplex_noise_schedule, add_simplex_noise

class PortfolioDiffusionDataset(Dataset):
    """
    Generate training data for hybrid denoiser from historical returns.
    """
    
    def __init__(self, returns_array, lookback=60, forward=20, timesteps=1000):
        """
        Args:
            returns_array: Price relatives [T, n_assets]
            lookback: History window for features
            forward: Forward window to compute optimal portfolio
            timesteps: Total diffusion timesteps
        """
        self.returns = torch.tensor(returns_array, dtype=torch.float32)
        self.lookback = lookback
        self.forward = forward
        self.timesteps = timesteps
        
        # Precompute noise schedules
        self.beta_schedule = cosine_beta_schedule(timesteps)
        self.alpha_bar = (1 - self.beta_schedule).cumprod(dim=0)
        self.simplex_alphas = simplex_noise_schedule(timesteps)
        
    def compute_optimal_portfolio(self, forward_returns):
        """
        Compute Kelly-optimal (log-optimal) portfolio.
        """
        log_returns = torch.log(forward_returns + 1e-8)
        mean_log_return = log_returns.mean(dim=0)
        cov_log_return = torch.cov(log_returns.T)
        
        try:
            # Simplified: proportional to inv(Σ) @ μ
            inv_cov = torch.linalg.inv(
                cov_log_return + 1e-6 * torch.eye(len(mean_log_return))
            )
            b_optimal = inv_cov @ mean_log_return
            b_optimal = torch.clamp(b_optimal, min=0)
            b_optimal = b_optimal / (b_optimal.sum() + 1e-8)
        except:
            # Fallback to equal weights
            b_optimal = torch.ones(len(mean_log_return)) / len(mean_log_return)
            
        return b_optimal
    
    def compute_market_state(self, returns_history):
        """
        Extract market state features from returns history.
        """
        # Basic features
        mean_ret = returns_history.mean(dim=0)
        std_ret = returns_history.std(dim=0)
        
        # Momentum (recent vs older)
        if len(returns_history) >= 40:
            momentum = returns_history[-20:].mean(dim=0) - returns_history[:-20].mean(dim=0)
        else:
            momentum = torch.zeros_like(mean_ret)
        
        # Correlation structure
        if len(returns_history) > 1:
            corr_matrix = torch.corrcoef(returns_history.T)
            upper_tri = torch.triu(corr_matrix, diagonal=1)
            avg_corr = upper_tri[upper_tri != 0].mean() if (upper_tri != 0).any() else torch.tensor(0.0)
        else:
            avg_corr = torch.tensor(0.0)
        
        # Volatility of volatility
        if len(returns_history) >= 10:
            rolling_vol = returns_history.unfold(0, 5, 1).std(dim=-1)
            vol_of_vol = rolling_vol.std(dim=0).mean()
        else:
            vol_of_vol = torch.tensor(0.0)
        
        z = torch.cat([mean_ret, std_ret, momentum, torch.tensor([avg_corr, vol_of_vol])])
        return z
    
    def __getitem__(self, idx):
        """Generate training sample with noise at random timestep."""
        start = idx
        end = start + self.lookback
        forward_end = end + self.forward
        
        returns_history = self.returns[start:end]
        forward_returns = self.returns[end:forward_end]
        
        # Compute clean targets
        z_0 = self.compute_market_state(returns_history)
        b_0 = self.compute_optimal_portfolio(forward_returns)
        
        # Sample random timestep
        t = torch.randint(0, self.timesteps, (1,)).item()
        
        # Add Gaussian noise to market state
        alpha_bar_t = self.alpha_bar[t]
        sigma_t = torch.sqrt(1 - alpha_bar_t)
        z_t = z_0 + sigma_t * torch.randn_like(z_0)
        
        # Add simplex noise to portfolio
        alpha_t = self.simplex_alphas[t]
        b_t = add_simplex_noise(b_0.unsqueeze(0), alpha_t, method='dirichlet').squeeze(0)
        
        return {
            'z_t': z_t,
            'b_t': b_t,
            'z_0': z_0,
            'b_0': b_0,
            'returns_history': returns_history,
            't': torch.tensor([t], dtype=torch.float32) / self.timesteps,
            'alpha_t': alpha_t,
            'sigma_t': sigma_t
        }
    
    def __len__(self):
        return len(self.returns) - self.lookback - self.forward
    
    