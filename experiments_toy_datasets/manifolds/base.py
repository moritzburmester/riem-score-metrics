from __future__ import annotations
from typing import Callable 
import numpy as np 
import torch 
from stochman.manifold import Manifold 

# code adopted from https://github.com/VictorBoutin/RiemannEBM/blob/main/tutorial/geodesics.ipynb

class BaseManifold(Manifold):

    # metrics take points (N, D) and return G (N, D, D)
    # BaseManifold wraps into stochman.Manifold interface
    # e.g. (B, N, D) input for stochman spline solver

    def __init__(self, metric_fn, *, name=''):
        super().__init__()
        self.metric_fn = metric_fn
        self.name = name

    def metric(self, points, *args, **kwargs):
        if points.ndim == 3:
            B, N, D = points.shape
            G = self.metric_fn(points.reshape(B * N, D))
            return G.reshape(B, N, D, D)
        return self.metric_fn(points)
    
def _eye(N, D, device, dtype):
    return torch.eye(D, device=device, dtype=dtype).expand(N, D, D)

def diag_to_metric(diag_fn: Callable):

    # converts h(x) (N, D) diagonal output into G(x) (N, D, D)

    def _G(x):
        h=diag_fn(x)
        N, D = x.shape
        G = torch.zeros(N, D, D, device=x.device, dtype=x.dtype)
        idx = torch.arange(D, device=x.device)
        G[:, idx, idx] = h
        return G
    return _G 

def linear_normalization(mini, maxi, target_max, target_min):
    alpha = (target_max - target_min)/(maxi - mini)
    beta = target_min - alpha*mini
    return alpha, beta

def _det_summary(G):
    # det(G)^(1/D) 
    D = G.shape[-1]
    return torch.det(G).clamp(min=1e-30).pow(1.0 / D)

def _trace_summary(G):
    # mean eigenvalue 
    return G.diagonal(dim1=-2, dim2=-1).mean(-1)

def normalize_metric(metric_fn, reference_points,
                     target_min=1.0, target_max=1000.0,
                     summary='trace', log_scale=False):
    summary_fn = _summaries[summary] if isinstance(summary, str) else summary

    with torch.no_grad():
        G = metric_fn(reference_points)
        s_raw = summary_fn(G).clamp(min=1e-30)
        s_cal = torch.log(s_raw) if log_scale else s_raw
        scale_min = float(s_cal.min())
        scale_max = float(s_cal.max())

    alpha, beta = linear_normalization(scale_min, scale_max, target_max, target_min)

    def normalized(x):
        G = metric_fn(x)
        s_raw = summary_fn(G).clamp(min=1e-30)
        if log_scale:
            target = alpha * torch.log(s_raw) + beta       
            ratio = target / s_raw                          
        else:
            ratio = alpha + beta / s_raw
        return ratio[:, None, None] * G

    return normalized, {
        'alpha': float(alpha), 'beta': float(beta),
        'scale_min_orig': scale_min, 'scale_max_orig': scale_max,
        'target_min': target_min, 'target_max': target_max,
        'summary': summary if isinstance(summary, str) else 'custom',
        'log_scale': log_scale,
    }


_summaries = {'trace': _trace_summary, 'det': _det_summary}
