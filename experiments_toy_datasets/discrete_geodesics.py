import numpy as np 
import torch 

# this file contains the optimization loop for the discrete geodesic optimization for the toy datasets
# corresponds to Algorithm 2: Discrete Energy Minimization in the thesis 

def discrete_geodesic_with_init(manifold, z0, z1, z_init, 
                                n_iter=5000, lr=5e-3, device='cpu', verbose_tag=None):

    z0 = z0.to(device).flatten() # shape: d, 
    z1 = z1.to(device).flatten()
    z_init = z_init.to(device) # shape: N, d 

    n_points = z_init.shape[0]
    dt = 1.0/ (n_points -1)
    z_i = z_init[1:-1].clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([z_i], lr=lr)

    history = []
    for ep in range(n_iter):
        optimizer.zero_grad()
        z_t = torch.cat([z0.unsqueeze(0), z_i, z1.unsqueeze(0)], dim=0)

        z_dot = (z_t[1:] - z_t[:-1]) / dt # finite differences

        G = manifold.metric(z_t[:-1])
        e = torch.einsum('ni, nij, nj->n', z_dot, G, z_dot)

        loss = (0.5 * e * dt).sum()

        loss.backward()
        optimizer.step()

        history.append(float(loss.item()))

        if verbose_tag and (ep == 0 or (ep + 1) % 500 == 0):
            print(f'{verbose_tag}, ep={ep}, loss={float(loss.item())}')

    with torch.no_grad():
        z_t = torch.cat([z0.unsqueeze(0), z_i, z1.unsqueeze(0)], dim=0)
    
    return z_t.detach().cpu(), history

def discrete_geodesic_with_init_symmetric(manifold, z0, z1, z_init,
                                n_iter=5000, lr=1e-3, device='cpu', verbose_tag=None):

    # symmetrized version (what Bethune et al uses in their paper)

    z0 = z0.to(device).flatten()
    z1 = z1.to(device).flatten()
    z_init = z_init.to(device)
    n_points = z_init.shape[0]
    dt = 1.0 / (n_points - 1)

    z_i = z_init[1:-1].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z_i], lr=lr)

    history = []
    for ep in range(n_iter):
        optimizer.zero_grad()
        z_t = torch.cat([z0[None], z_i, z1[None]], dim=0)

        z_dot = (z_t[1:] - z_t[:-1]) / dt
        G_fwd = manifold.metric(z_t[:-1])
        e_fwd = torch.einsum('ni,nij,nj->n', z_dot, G_fwd, z_dot)

        z_inv = torch.flip(z_t, dims=[0])
        z_dot_inv = (z_inv[1:] - z_inv[:-1]) / dt
        G_rev = manifold.metric(z_inv[:-1])
        e_rev = torch.einsum('ni,nij,nj->n', z_dot_inv, G_rev, z_dot_inv)

        loss = (0.5 * (e_fwd + e_rev) * dt).sum()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

        if verbose_tag and (ep == 0 or (ep + 1) % 500 == 0):
            print(f'{verbose_tag}, ep={ep}, loss={float(loss.item())}')

    with torch.no_grad():
        z_t = torch.cat([z0[None], z_i, z1[None]], dim=0)
    return z_t.detach().cpu(), history