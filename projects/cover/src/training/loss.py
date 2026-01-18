import torch

def hybrid_denoising_loss(model, batch, lambda_market=1.0, lambda_portfolio=1.0, lambda_coupling=0.5):
    """
    Combined loss for hybrid denoiser.
    
    L = λ_m ||ẑ₀ - z₀||² + λ_p ||b̂₀ - b₀||² + λ_c ||f(ẑ₀) - b̂₀||²
    """
    z_t = batch['z_t']
    b_t = batch['b_t']
    z_0_target = batch['z_0']
    b_0_target = batch['b_0']
    returns_history = batch['returns_history']
    t = batch['t']
    
    # Forward pass
    z_0_pred, b_0_pred = model(z_t, b_t, returns_history, t)
    
    # Market state denoising loss
    loss_market = torch.mean((z_0_pred - z_0_target) ** 2)
    
    # Portfolio denoising loss
    loss_portfolio = torch.mean((b_0_pred - b_0_target) ** 2)
    
    # Coupling consistency loss
    t_emb = model.time_embed(t)
    b_from_market = model.state_to_portfolio(
        torch.cat([z_0_pred.detach(), t_emb], dim=-1)
    )
    b_from_market = torch.softmax(b_from_market, dim=-1)
    loss_coupling = torch.mean((b_0_pred - b_from_market) ** 2)
    
    # Combined loss
    total_loss = (
        lambda_market * loss_market +
        lambda_portfolio * loss_portfolio +
        lambda_coupling * loss_coupling
    )
    
    return total_loss, {
        'loss_market': loss_market.item(),
        'loss_portfolio': loss_portfolio.item(),
        'loss_coupling': loss_coupling.item()
    }

