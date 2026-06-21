import numpy as np 
from scipy.special import logsumexp 
import torch

# this file includes analytic gmm densities for the datasets used 

def _gmm_log_density(x, means, sigma, weights=None):
    N, D = x.shape # number of datapoints, ambient dimension
    K = means.shape[0] # number of components 
    diff = x[:, None, :] - means[None, :, :] # x - mu term 
    mahal = (diff ** 2).sum(-1) / (sigma ** 2) # squared mahalanobis distance
    log_norm = -0.5 * D * np.log(2 * np.pi * sigma ** 2) # first term 
    log_w = np.log(weights if weights is not None else np.ones(K) / K) # log weights
    return logsumexp(log_w[None] + log_norm - 0.5 * mahal, axis=1) # log sum trick 

def gmm_logp(x, means, sigma, weights=None):
    N, D = x.shape; K = means.shape[0]
    diff = x[:, None, :] - means[None, :, :]
    sq = (diff ** 2).sum(-1) / (sigma ** 2)
    log_norm = -0.5 * D * torch.log(torch.tensor(2 * torch.pi * sigma ** 2,
                                                 device=x.device, dtype=x.dtype))
    if weights is None:
        log_w = -torch.log(torch.tensor(float(K), device=x.device, dtype=x.dtype))
    else:
        log_w = torch.log(weights.to(x.device).clamp(min=1e-30))
    return torch.logsumexp(log_w + log_norm - 0.5 * sq, dim=1)

def log_density_circle(x, meta, n_components=1000):
    # helper function 
    X, means, sigma = get_gmm_components('circle', meta, x, n_components)
    return __gmm_log_density(x, means, sigma)

def _gmm_responsibilities(x, means, sigma, weights=None):
    N, D = x.shape; K = means.shape[0]
    diff = x[:, None, :] - means[None, :, :]
    sq = (diff ** 2).sum(-1)
    var = sigma ** 2
    log_phi = -0.5 * sq/var 

    if weights is None: 
        log_w = -torch.log(torch.tensor(float(K), device=x.device))
        log_terms = log_phi + log_w 
    else: 
        log_w = torch.log(weights.to(x.device).clamp(min=1e-30))
        log_terms = log_phi + log_w
    log_norm = torch.logsumexp(log_terms, dim=1, keepdim=True)
    r = torch.exp(log_terms - log_norm)
    s_k = -diff / var
    s = (r.unsqueeze(-1) *  s_k).sum(dim=1)
    return r, s_k, s

def gmm_score(x, means, sigma, weights=None):
    _, _, s = _gmm_responsibilities(x, means, sigma, weights)
    return s 

def gmm_hessian(x, means, sigma, weights=None):
    r, s_k, s = _gmm_responsibilities(x, means, sigma, weights)
    N, D = x.shape
    var = sigma ** 2
    sk_outer = s_k.unsqueeze(-1) * s_k.unsqueeze(-2)
    term1 = (r.unsqueeze(-1).unsqueeze(-1) * sk_outer).sum(dim=1)
    ss = s.unsqueeze(-1) * s.unsqueeze(-2)
    I = torch.eye(D, device=x.device, dtype=x.dtype).expand(N, D, D)
    return term1 - ss - I/var

def get_gmm_components(name, meta, n_components=200):

    if name == '1-sphere':
        n_components = 1500
        radius = meta['radius_post']
        center = np.array(meta['center_post'])
        sigma = meta['noise'] / meta['scale_pre']
        theta = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
        means = center + radius * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        weights = None
    elif name in ('ucg', 'wcg'):
        K = meta['n_components']
        R_pre = meta['radius_pre']
        angles = np.linspace(0, np.pi, K)
        means_pre = R_pre * np.stack([np.cos(angles), np.sin(angles)], -1)
        means = (means_pre - np.array(meta['center_pre'])) / meta['scale_pre']
        sigma = meta['noise'] / meta['scale_pre']
        if name == 'wcg':
            lin = np.concatenate([np.linspace(1, 30, 90), np.ones(11) * 30])
            weights = np.concatenate([lin, lin[1:-1][::-1]])
            weights = weights / weights.sum()
        else:
            weights = None
    elif name == 'spiral':
        n_components = 1500
        t_min, t_max = meta['t_range']
        sigma  = meta['noise'] / meta['scale_pre']
        t      = np.linspace(t_min, t_max, n_components)
        spiral = np.stack([t * np.cos(t), t * np.sin(t)], -1)
        means  = (spiral - np.array(meta['center_pre'])) / meta['scale_pre']
        weights = None
    
    elif name == 'two_moons':
        n_components = 1500
        sigma = meta['noise'] / meta['scale_pre']

        n1 = n_components // 2
        n2 = n_components - n1

        t1 = np.linspace(0, np.pi, n1)
        t2 = np.linspace(0, np.pi, n2)

        moon1 = np.stack([np.cos(t1), np.sin(t1)], axis=-1)

        moon2 = np.stack([
            1.0 - np.cos(t2),
            1.0 - np.sin(t2) - 0.5
        ], axis=-1)

        means_pre = np.concatenate([moon1, moon2], axis=0)

        means = (
            means_pre - np.array(meta['center_pre'])
        ) / meta['scale_pre']

        weights = None
    elif name == 's-curve':
        n_components = 1500

        t_min, t_max = meta['t_range']

        sigma = meta['noise'] / meta['scale_pre']

        t = np.linspace(t_min, t_max, n_components)

        s_curve = np.stack([
            np.sign(t) * (np.cos(t) - 1),
            np.sin(t)
        ], axis=-1)

        means = (
            s_curve - np.array(meta['center_pre'])
        ) / meta['scale_pre']

        weights = None

    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    means_t = torch.tensor(means, dtype=torch.float32)
    weights_t = (torch.tensor(weights, dtype=torch.float32)
                 if weights is not None else None)
    return means_t, float(sigma), weights_t


class AnalyticDeriv: 
    # same interface has DiffusionDeriv class, analytic score/hessian of GMMs instead of jacrev

    def __init__(self, means, sigma, weights=None, t=None):
        self.means = means
        self.sigma = sigma
        self.weights = weights
        self.t = t # not used for analytical score
    
    def to(self, dtype=None, device=None):
        new_means = self.means.to(dtype=dtype, device=device)
    
        new_weights = (self.weights.to(dtype=dtype, device=device)
                    if self.weights is not None
                    and (dtype is not None or device is not None)
                    else self.weights)
        return AnalyticDeriv(new_means, self.sigma, new_weights, self.t)

    def logp(self, x):
        m = self.means.to(x.device)
        w = self.weights.to(x.device) if self.weights is not None else None
        return gmm_logp(x, m, self.sigma, w)

    def score(self, x):
        m = self.means.to(x.device)
        w = self.weights.to(x.device) if self.weights is not None else None
        return gmm_score(x, m, self.sigma, w)
        
    def hessian(self, x):
        m = self.means.to(x.device)
        w = self.weights.to(x.device) if self.weights is not None else None
        return gmm_hessian(x, m, self.sigma, w)