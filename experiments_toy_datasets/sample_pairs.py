import numpy as np


def _circle(meta, n_pairs=5):
    center = np.asarray(meta['center_post'], dtype=np.float64)
    r = float(meta['radius_post'])

    pairs = [(
        np.array([-1.5,  0.5], dtype=np.float64),
        np.array([ 1.5, -0.5], dtype=np.float64),
    )]
    remaining = max(0, n_pairs - 1)
    if remaining:
        base = np.linspace(0.0, np.pi, remaining, endpoint=False)
        seps = np.linspace(2 * np.pi / 3, 11 * np.pi / 12, remaining)
        for a, sep in zip(base, seps):
            b = a + sep
            z0 = center + r * np.array([np.cos(a), np.sin(a)])
            z1 = center + r * np.array([np.cos(b), np.sin(b)])
            pairs.append((z0, z1))
    return pairs[:n_pairs]


def _semicircle(meta, n_pairs=5):
    R = float(meta['radius_pre'])
    n = 1500
    ang = np.linspace(0, np.pi, n)
    means_pre = R * np.stack([np.cos(ang), np.sin(ang)], -1)
    means = (means_pre - np.asarray(meta['center_pre'])) / meta['scale_pre']

    pairs = [(
        np.array([-2.0, -1.5], dtype=np.float64),
        np.array([ 2.0, -1.5], dtype=np.float64),
    )]
    remaining = max(0, n_pairs - 1)
    if remaining:
        lo = np.linspace(0.02, 0.30, remaining)
        hi = np.linspace(0.70, 0.98, remaining)
        for s, e in zip(lo, hi):
            pairs.append((means[int(s * (n - 1))], means[int(e * (n - 1))]))
    return pairs[:n_pairs]


def _spiral(meta, n_pairs=5):
    t_min, t_max = meta['t_range']
    n = 1500
    t = np.linspace(t_min, t_max, n)
    spiral_pre = np.stack([t * np.cos(t), t * np.sin(t)], -1)
    means = (spiral_pre - np.asarray(meta['center_pre'])) / meta['scale_pre']

    pairs = [(
        np.array([-0.25, -0.75], dtype=np.float64),
        np.array([-0.25,  2.0 ], dtype=np.float64),
    )]
    remaining = max(0, n_pairs - 1)
    if remaining:
        frac = 0.32
        starts = np.linspace(0.02, 1.0 - frac - 0.02, remaining)
        for s in starts:
            i = int(s * (n - 1))
            j = int(min(0.99, s + frac) * (n - 1))
            pairs.append((means[i], means[j]))
    return pairs[:n_pairs]

def _two_moons(meta, n_pairs=5):
    # both endpoints on the same moon 
    n_components = 1500
    n1 = n_components // 2
    n2 = n_components - n1
    t1 = np.linspace(0, np.pi, n1)
    t2 = np.linspace(0, np.pi, n2)
    moon1 = np.stack([np.cos(t1), np.sin(t1)], -1)
    moon2 = np.stack([1.0 - np.cos(t2), 1.0 - np.sin(t2) - 0.5], -1)
    means_pre = np.concatenate([moon1, moon2], 0)
    means = (means_pre - np.asarray(meta['center_pre'])) / meta['scale_pre']

    plan = [
        ('m1', 0.02, 0.98),
        ('m1', 0.05, 0.55),
        ('m1', 0.45, 0.95),
        ('m2', 0.02, 0.98),
        ('m2', 0.10, 0.60),
    ][:n_pairs]
    pairs = []
    for which, s, e in plan:
        if which == 'm1':
            i, j = int(s * (n1 - 1)), int(e * (n1 - 1))
        else:
            i = n1 + int(s * (n2 - 1))
            j = n1 + int(e * (n2 - 1))
        pairs.append((means[i], means[j]))
    return pairs

def _s_curve(meta, n_pairs=5):
    t_min, t_max = meta['t_range']
    n = 1500
    t = np.linspace(t_min, t_max, n)
    s_pre = np.stack([np.sign(t) * (np.cos(t) - 1), np.sin(t)], -1)
    means = (s_pre - np.asarray(meta['center_pre'])) / meta['scale_pre']
    plan = [
        (0.02, 0.98),
        (0.05, 0.55),
        (0.45, 0.95),
        (0.10, 0.75),
        (0.25, 0.90),
    ][:n_pairs]
    pairs = []
    for s, e in plan:
        pairs.append((means[int(s * (n - 1))], means[int(e * (n - 1))]))
    return pairs


pair_samplers = {
    '1-sphere': _circle,
    'ucg':      _semicircle,
    'wcg':      _semicircle,
    'spiral':   _spiral,
    'two_moons': _two_moons,
    's-curve':  _s_curve,
}

def get_pairs(dataset_name, meta, *, n_pairs=5, off_manifold=False,
              noise=0.3, seed=0):
    pairs = pair_samplers[dataset_name](meta, n_pairs=n_pairs)
    if off_manifold:
        rng = np.random.default_rng(seed)
        perturbed = []
        for z0, z1 in pairs:
            z0p = z0 + noise * rng.standard_normal(z0.shape)
            z1p = z1 + noise * rng.standard_normal(z1.shape)
            perturbed.append((z0p.astype(np.float32), z1p.astype(np.float32)))
        return perturbed
    return [(z0.astype(np.float32), z1.astype(np.float32)) for z0, z1 in pairs]