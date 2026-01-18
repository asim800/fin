import torch
import torch.nn as nn
from .components import SinusoidalPositionEmbedding, project_to_simplex

class HybridPortfolioDenoiser(nn.Module):
    """
    Joint denoiser for market state and portfolio allocation.
    
    Combines:
    - Gaussian Tweedie (market state denoising)
    - Portfolio Tweedie (allocation denoising)
    - Learned coupling (consistency enforcement)
    """
    
    def __init__(self, n_assets, d_market, d_hidden=128, n_lstm_layers=2):
        super().__init__()
        self.n_assets = n_assets
        self.d_market = d_market
        self.d_hidden = d_hidden
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        
        # Market state denoiser (Gaussian Tweedie component)
        self.market_denoiser = nn.Sequential(
            nn.Linear(d_market + d_hidden, d_hidden),
            nn.SiLU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_market)
        )
        
        # Returns history encoder
        self.returns_encoder = nn.LSTM(
            input_size=n_assets,
            hidden_size=d_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=0.1 if n_lstm_layers > 1 else 0
        )
        
        # Portfolio denoiser (Portfolio Tweedie component)
        self.portfolio_denoiser = nn.Sequential(
            nn.Linear(n_assets + d_hidden + d_hidden, d_hidden),
            nn.SiLU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, n_assets)
        )
        
        # Coupling network: market state → portfolio
        self.state_to_portfolio = nn.Sequential(
            nn.Linear(d_market + d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, n_assets)
        )
        
        # Learned coupling strength
        self.coupling_strength = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, z_t, b_t, returns_history, t):
        """
        Joint denoising with coupling.
        
        Args:
            z_t: Noisy market state [batch, d_market]
            b_t: Noisy portfolio [batch, n_assets]
            returns_history: Past returns [batch, lookback, n_assets]
            t: Timestep [batch, 1]
        
        Returns:
            z_0_pred: Denoised market state [batch, d_market]
            b_0_pred: Denoised portfolio [batch, n_assets]
        """
        batch_size = z_t.shape[0]
        
        # 1. Time embedding
        t_emb = self.time_embed(t)
        
        # 2. Encode returns history
        returns_features, _ = self.returns_encoder(returns_history)
        returns_features = returns_features[:, -1, :]  # Last hidden state
        
        # 3. Denoise market state (Gaussian Tweedie)
        market_input = torch.cat([z_t, t_emb], dim=-1)
        z_0_pred = self.market_denoiser(market_input)
        
        # 4. Denoise portfolio (Portfolio Tweedie)
        portfolio_input = torch.cat([b_t, returns_features, t_emb], dim=-1)
        portfolio_delta = self.portfolio_denoiser(portfolio_input)
        b_0_uncoupled = b_t + portfolio_delta
        
        # 5. Coupling: enforce consistency
        b_consistent = self.state_to_portfolio(
            torch.cat([z_0_pred, t_emb], dim=-1)
        )
        
        # Weighted combination
        beta = torch.sigmoid(self.coupling_strength)
        b_0_pred = (1 - beta) * b_0_uncoupled + beta * b_consistent
        
        # 6. Project to simplex
        b_0_pred = project_to_simplex(b_0_pred)
        
        return z_0_pred, b_0_pred
    

    