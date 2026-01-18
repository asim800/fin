import torch

def denoise_portfolio(model, returns_history, n_steps=50, device='cpu'):
    """
    Generate portfolio allocation using iterative denoising.
    
    Args:
        model: Trained HybridPortfolioDenoiser
        returns_history: Recent returns [lookback, n_assets]
        n_steps: Number of denoising steps
        device: torch device
    
    Returns:
        b_final: Denoised portfolio [n_assets]
        z_final: Denoised market state [d_market]
    """
    model.eval()
    n_assets = model.n_assets
    d_market = model.d_market
    
    # Initialize from noise/uniform
    z_t = torch.randn(1, d_market).to(device)
    b_t = torch.ones(1, n_assets).to(device) / n_assets
    
    returns_history = returns_history.unsqueeze(0).to(device)
    
    # Denoising timesteps
    timesteps = torch.linspace(1.0, 0.0, n_steps)
    
    with torch.no_grad():
        for i, t in enumerate(timesteps[:-1]):
            t_batch = t.unsqueeze(0).unsqueeze(0).to(device)
            
            # Denoise one step
            z_0_pred, b_0_pred = model(z_t, b_t, returns_history, t_batch)
            
            if i < n_steps - 1:
                # DDIM-style update
                t_next = timesteps[i + 1]
                
                # Market state (Gaussian)
                alpha_t = 1 - t**2
                alpha_t_next = 1 - t_next**2
                
                noise_pred = (z_t - torch.sqrt(alpha_t) * z_0_pred) / torch.sqrt(1 - alpha_t + 1e-8)
                z_t = torch.sqrt(alpha_t_next) * z_0_pred + torch.sqrt(1 - alpha_t_next) * noise_pred
                
                # Portfolio (move toward prediction)
                b_t = 0.8 * b_0_pred + 0.2 * b_t
                b_t = b_t / b_t.sum(dim=-1, keepdim=True)
            else:
                z_t = z_0_pred
                b_t = b_0_pred
    
    return b_t.squeeze(0).cpu().numpy(), z_t.squeeze(0).cpu().numpy()


