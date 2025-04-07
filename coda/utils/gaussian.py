import numpy as np
import torch

def sample_from_diag_gaussian(mu, sigma, n_samples=1, device='cuda', debug='True'):
    std = torch.sqrt(sigma)
    eps = torch.randn(n_samples, std.shape[0], std.shape[1]).to(device)
    if n_samples == 1:
        eps = eps.squeeze(0)
    if debug:
        print(mu.shape, std.shape, eps.shape)
    samples = mu + eps.std
    return samples
    