import torch

# lerp (slerp not used for toy data)

def lerp_curve(z0, z1, n):
    t = torch.linspace(0, 1, n, device=z0.device).view(-1, 1)
    return (1 - t) * z0 + t * z1
