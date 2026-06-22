import torch 
from .base import BaseManifold, _eye, normalize_metric, _trace_summary, _summaries
from .score_analytic import AnalyticDeriv 

_G_T_H = None
_G_T_s = None

def normalize_G(deriv, reference_points, *, noise_scale=None, n_noisy=1,
              include_clean=True):
    global _G_T_H, _G_T_s
    with torch.no_grad():
        parts = [reference_points] if include_clean else []
        if noise_scale is not None:
            for _ in range(n_noisy):
                parts.append(reference_points + noise_scale * torch.randn_like(reference_points))
        S = torch.cat(parts, 0) if len(parts) > 1 else parts[0]

        H = deriv.hessian(S)
        HtH = H @ H.transpose(-1, -2)
        sn2 = deriv.score(S).pow(2).sum(-1)
        I = _eye(S.shape[0], S.shape[1], S.device, S.dtype)
        dens = sn2[:, None, None] * I

        tH = _trace_summary(HtH).clamp(min=1e-30)
        tS = _trace_summary(dens).clamp(min=1e-30)
        _G_T_H = float(tH.mean())
        _G_T_s = float(tS.mean())
        postA = float((tH / _G_T_H).mean())
        postB = float((tS / _G_T_s).mean())

        nclean = reference_points.shape[0] if include_clean else 0
    return {'kind': 'G', 'T_H': _G_T_H, 'T_s': _G_T_s,
            'n_ref': int(S.shape[0]), 'n_clean': int(nclean),
            'noise_scale': (float(noise_scale) if noise_scale is not None else None),
            'post_norm_mean_termA': postA, 'post_norm_mean_termB': postB}

def _G(points, deriv, lam=0.1, normalize=True):

    H = deriv.hessian(points)
    HtH = H @ H.transpose(-1, -2) # HHT or HtH (since its symmetric)
    sn2 = deriv.score(points).pow(2).sum(-1)
    N, D = points.shape
    I = _eye(N, D, points.device, points.dtype)

    if normalize:
        if _G_T_H is None or _G_T_s is None:
            raise RuntimeError("call normalize_G(deriv, reference_points) before _G")
        T_H, T_s = _G_T_H, _G_T_s
    else:
        T_H = T_s = 1.0
    A = HtH / T_H 
    B = (sn2[:, None, None] * I) / T_s
    return (1 - lam) * A + lam * B 

def _G_quad(points, v, deriv, lam=0.1, normalize=True):

    H = deriv.hessian(points)                      # N, D, D
    Hv = torch.einsum('nij,nj->ni', H, v)          # N, D
    sn2 = deriv.score(points).pow(2).sum(-1)       # N,

    if normalize:
        if _G_T_H is None or _G_T_s is None:
            raise RuntimeError("call normalize_G(deriv, reference_points) before _G_quad")
        T_H, T_s = _G_T_H, _G_T_s
    else:
        T_H = T_s = 1.0

    A = Hv.pow(2).sum(-1) / T_H                     
    B = sn2 * v.pow(2).sum(-1) / T_s                
    return (1 - lam) * A + lam * B

## metrics from other literature 

def _INVP(points, deriv, logp_floor=-15.0):
    # inverse-probability metric: G = (1/p)  * I 
    N, D = points.shape
    I = _eye(N, D, points.device, points.dtype)
    logp = deriv.logp(points).clamp(min=logp_floor)
    scale = torch.exp(-logp)
    return scale[:, None, None] * I
    

def _SAI(points, deriv, tau=1e-3):
    # G = HTH
    H = deriv.hessian(points)
    HtH = H @ H.transpose(-1, -2)
    N, D = points.shape
    return HtH + tau * _eye(N, D, points.device, points.dtype)


def _AZE(points, deriv, lam=1.0):
    # Azeglio (2025) 
    # G = I + lam * ssT
    # Penalizes normal direction movement
    s = deriv.score(points)
    sst = s.unsqueeze(-1) * s.unsqueeze(-2)
    N, D = points.shape
    return _eye(N, D, points.device, points.dtype) + lam * sst


def _PER(points, deriv):
    # Perone (2024)
    # G = I − ssT / (1 +||s||^2)
    # Penalizes tangent movement
    s = deriv.score(points)
    sst = s.unsqueeze(-1) * s.unsqueeze(-2)
    s_norm2 = s.pow(2).sum(-1)                                
    denom = (1.0 + s_norm2).unsqueeze(-1).unsqueeze(-1)
    N, D = points.shape
    return _eye(N, D, points.device, points.dtype) - sst / denom


def _build(metric_fn, name, deriv, reference_points,
           target_min, target_max, summary='trace',
           log_scale=False, normalize=True, **hparams):

    raw = lambda x: metric_fn(x, deriv, **hparams)

    if normalize:
        normalized, stats = normalize_metric(raw, reference_points,
                                             target_min=target_min,
                                             target_max=target_max,
                                             summary=summary, log_scale=log_scale)
        return BaseManifold(normalized, name=name), {'kind': name, **hparams, **stats}

    with torch.no_grad():
        G = raw(reference_points)
        summary_fn = _summaries[summary] if isinstance(summary, str) else summary
        s = summary_fn(G).clamp(min=1e-30)
        stats = {
            'kind': name, **hparams,
            'normalized': False,
            'summary': summary,
            'scale_min_orig': float(s.min()),
            'scale_max_orig': float(s.max()),
        }
    return BaseManifold(raw, name=name), stats

def build_SAI(deriv, reference_points, *, tau=0,
              target_min=1.0, target_max=1000.0, normalize=False):
    return _build(_SAI, 'SAI', deriv, reference_points, target_min, target_max,
                  summary='trace', normalize=normalize, tau=tau)

def build_PER(deriv, reference_points, *,
              target_min=1.0, target_max=1000.0, normalize=False):
    return _build(_PER, 'PER', deriv, reference_points, target_min, target_max,
                  summary='trace', normalize=normalize)

def build_AZE(deriv, reference_points, *, lam=1.0,
              target_min=1.0, target_max=1000.0, normalize=False):
    return _build(_AZE, 'AZE', deriv, reference_points, target_min, target_max,
                  summary='trace', normalize=normalize, lam=lam)

def build_G(deriv, reference_points, *,
             target_min=1.0, target_max=1000.0, normalize=False):
    return _build(_G, 'G', deriv, reference_points, target_min, target_max,
                  summary='trace', normalize=normalize)

def build_INVP(deriv, reference_points, *,
             target_min=1.0, target_max=1000.0, normalize=True):
    return _build(_INVP, 'INVP', deriv, reference_points, target_min, target_max,
                  summary='trace', normalize=normalize)