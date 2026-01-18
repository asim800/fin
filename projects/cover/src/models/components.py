import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbedding(nn.Module):
    """Standard sinusoidal embedding for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        # Handle both [batch] and [batch, 1] inputs
        if t.dim() > 1:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

def project_to_simplex(x, eps=1e-8):
    """Project vector to probability simplex"""
    x = torch.clamp(x, min=0)
    return x / (x.sum(dim=-1, keepdim=True) + eps)


