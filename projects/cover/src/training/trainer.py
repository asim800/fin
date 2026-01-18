import torch
from torch.utils.data import DataLoader
from .loss import hybrid_denoising_loss

def train_hybrid_denoiser(
    dataset,
    model,
    n_epochs=100,
    batch_size=32,
    lr=1e-4,
    device='cpu'
):
    """
    Train the hybrid denoising model.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model = model.to(device)
    model.train()
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Compute loss
            loss, loss_components = hybrid_denoising_loss(model, batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    return model

