# metrics/data_driven.py
from __future__ import annotations

from .base import BaseManifold, _eye, linear_normalization

# RBF, LAND, and EBM implemented as in Bethune et al. 

class h_diag_RBF(nn.Module):
    def __init__(self, n_centers, data_size=2,
                 data_to_fit_ambiant=None, data_to_fit_latent=None, kappa=1.0):
        super().__init__()
        self.K = n_centers
        self.data_size = data_size
        self.kappa = kappa
        self.W = nn.Parameter(torch.rand(self.K, 1))

        sigmas = np.ones((self.K, data_size))
        if data_to_fit_ambiant is not None and data_to_fit_latent is not None:
            a = data_to_fit_ambiant.detach().cpu().numpy()
            l = data_to_fit_latent.detach().cpu().numpy()
            km = KMeans(n_clusters=self.K, n_init=10).fit(a)
            clusters = km.cluster_centers_
            labels = km.labels_
            self.register_buffer('C', torch.tensor(clusters, dtype=torch.float32))
            for k in range(self.K):
                pts = l[labels == k]
                if pts.shape[0]:
                    var = ((pts - clusters[k]) ** 2).mean(axis=0)
                    sigmas[k, :] = np.sqrt(np.maximum(var, 1e-12))
        else:
            self.register_buffer('C', torch.zeros(self.K, data_size))

        lbda = torch.tensor(0.5 / (self.kappa * sigmas) ** 2, dtype=torch.float32)
        self.register_buffer('lamda', lbda)

    def h(self, x):
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        dist2 = torch.cdist(x, self.C) ** 2                     # (B, K)
        phi = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])  # (B, K, D)
        return phi.sum(dim=1)                                   # (B, D)


class h_diag_Land(nn.Module):
    def __init__(self, reference_sample, gamma=0.2,
                 max_ref=None, chunk=4096, seed=0):
        super().__init__()
        ref = reference_sample.detach().clone()
        if max_ref is not None and ref.shape[0] > max_ref:
            g = torch.Generator(device='cpu').manual_seed(seed)
            sel = torch.randperm(ref.shape[0], generator=g)[:max_ref]
            ref = ref[sel.to(ref.device)]
        self.register_buffer('ref', ref)
        self.gamma = float(gamma)
        self.chunk = int(chunk)

    def h(self, x):
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        inv2g2 = 1.0 / (2.0 * self.gamma ** 2)
        outs = []
        for s in range(0, x.shape[0], self.chunk):
            xb = x[s:s + self.chunk]                              # (b, D)
            diff2 = (xb[:, None, :] - self.ref[None, :, :]) ** 2  # (b, N_ref, D)
            sq = diff2.sum(-1)                                    # (b, N_ref)
            w = torch.exp(-sq * inv2g2)                           # (b, N_ref)
            outs.append(torch.einsum('bn,bnd->bd', w, diff2))     # (b, D)
        return torch.cat(outs, dim=0)


def normalize_diag(h_fn, reference_points, mini=1e-3, maxi=1.0):
    with torch.no_grad():
        h = h_fn(reference_points)
        h_min = float(h.min())
        h_max = float(h.max())
    alpha, beta = linear_normalization(h_min, h_max, maxi, mini)
    return alpha, beta


def _diag_to_G(diag_vec):
    N, D = diag_vec.shape
    G = diag_vec.new_zeros(N, D, D)
    idx = torch.arange(D, device=diag_vec.device)
    G[:, idx, idx] = diag_vec
    return G

def build_RBF(reference_points, *, n_centers=30, kappa=1.0,
              mini=1e-3, maxi=1.0):
    device = reference_points.device
    head = h_diag_RBF(
        n_centers=n_centers, data_size=reference_points.shape[1],
        data_to_fit_ambiant=reference_points,
        data_to_fit_latent=reference_points, kappa=kappa,
    ).to(device)
    alpha, beta = normalize_diag(head.h, reference_points, mini=mini, maxi=maxi)

    def metric_fn(x):
        diag = 1.0 / (alpha * head.h(x) + beta).clamp(min=1e-30)
        return _diag_to_G(diag)

    return BaseManifold(metric_fn, name='RBF'), {
        'kind': 'RBF', 'alpha': float(alpha), 'beta': float(beta),
        'n_centers': n_centers, 'kappa': kappa, 'norm_target': [mini, maxi],
    }


def build_LAND(reference_points, *, gamma=1.0, mini=1e-3, maxi=1.0,
               max_ref=None, chunk=4096):
    device = reference_points.device
    head = h_diag_Land(reference_sample=reference_points, gamma=gamma,
                       max_ref=max_ref, chunk=chunk).to(device)
    alpha, beta = normalize_diag(head.h, reference_points, mini=mini, maxi=maxi)

    def metric_fn(x):
        diag = 1.0 / (alpha * head.h(x) + beta).clamp(min=1e-30)
        return _diag_to_G(diag)

    return BaseManifold(metric_fn, name='LAND'), {
        'kind': 'LAND', 'alpha': float(alpha), 'beta': float(beta),
        'gamma': gamma, 'norm_target': [mini, maxi],
        'max_ref': max_ref, 'chunk': chunk,
    }

def build_EBM_en(ebm, reference_points, *, target_min=0.0, target_max=1e3):
    """
    G(x) = (1 + alpha * E(x) + beta) * I,
    """
    ebm = ebm.eval()
    with torch.no_grad():
        en = ebm(reference_points).squeeze(-1)
        e_min = float(en.min())
        e_max = float(en.max())
    alpha, beta = linear_normalization(e_min, e_max, target_max, target_min)

    def metric_fn(x):
        e = ebm(x).squeeze(-1)
        scale = 1.0 + alpha * e + beta
        N, D = x.shape
        return scale[:, None, None] * _eye(N, D, x.device, x.dtype)

    return BaseManifold(metric_fn, name='EBM_en'), {
        'kind': 'EBM_en', 'alpha': float(alpha), 'beta': float(beta),
        'e_min': e_min, 'e_max': e_max,
        'target_min': target_min, 'target_max': target_max,
    }


