import torch
import numpy as np

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for Gaussian diffusion on market state.
    From "Improved Denoising Diffusion Probabilistic Models"
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def simplex_noise_schedule(timesteps, alpha_max=100.0, alpha_min=1.0):
    """
    Noise schedule for portfolio weights using Dirichlet concentration.
    High concentration (low noise) → Low concentration (high noise)
    """
    alphas = torch.logspace(
        np.log10(alpha_max), 
        np.log10(alpha_min), 
        timesteps
    )
    return alphas

def add_simplex_noise(b_clean, alpha_t, method='dirichlet'):
    """
    Add noise to portfolio weights while preserving simplex constraint.
    
    Args:
        b_clean: Clean portfolio weights [batch, n_assets]
        alpha_t: Concentration parameter (scalar or [batch])
        method: 'dirichlet', 'projected_gaussian', or 'multiplicative'
    
    Returns:
        b_noisy: Noisy portfolio weights [batch, n_assets]
    """
    if method == 'dirichlet':
        # Sample from Dirichlet(alpha_t * b_clean)
        concentration = alpha_t * b_clean
        concentration = torch.clamp(concentration, min=0.01)
        dist = torch.distributions.Dirichlet(concentration)
        b_noisy = dist.sample()
        
    elif method == 'projected_gaussian':
        # Add Gaussian noise and project to simplex
        sigma_t = 1.0 / torch.sqrt(alpha_t)
        noise = torch.randn_like(b_clean) * sigma_t
        b_noisy = b_clean + noise
        b_noisy = torch.clamp(b_noisy, min=0)
        b_noisy = b_noisy / b_noisy.sum(dim=-1, keepdim=True)
        
    elif method == 'multiplicative':
        # Multiplicative log-normal noise
        sigma_t = 1.0 / torch.sqrt(alpha_t)
        log_noise = torch.randn_like(b_clean) * sigma_t
        b_noisy = b_clean * torch.exp(log_noise)
        b_noisy = b_noisy / b_noisy.sum(dim=-1, keepdim=True)
        
    return b_noisy

