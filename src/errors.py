import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def reconstruction_errors_and_recons(model, X_tensor, device, batch_size=128):
    model.eval()
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    errors, recons_list, embeds = [], [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            recon, emb = model(xb)
            e = torch.mean((xb - recon)**2, dim=(1,2)).cpu().numpy()
            errors.append(e)
            recons_list.append(recon.cpu().numpy())
            embeds.append(emb.cpu().numpy())
    return np.concatenate(errors), np.concatenate(recons_list), np.concatenate(embeds)
