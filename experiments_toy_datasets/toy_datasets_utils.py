import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll, make_s_curve
from pathlib import Path
import json 

# this file contains dataset generators for the 2D toy datasets used in our experiments. 

def _normalize(X):

    # subtract mean and divide by overall std"

    X = X.astype(np.float32)
    mean = X.mean(0, keepdims=True)
    Xc = X - mean
    scale = float(Xc.std() + 1e-12)

    return Xc / scale, mean.flatten(), scale

def _fit_circle(X):

    # fit circle by mean and average distance to mean

    center = X.mean(0)
    radius = float(np.linalg.norm(X - center, axis=1).mean())
    return center, radius

def generate_circle(n = 10000, radius = 1.0, noise = 0.03):

    # thin full-circle in 2d

    rng = np.random.default_rng(0)
    theta = 2 * np.pi * rng.uniform(size=n)
    X = radius * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    X = X + noise * rng.normal(size=(n, 2))
    Xn, mean, scale = _normalize(X)
    geom_center_pre = np.zeros(2)
    center_post = (geom_center_pre - mean) / scale
    radius_post = radius / scale

    # optional to do: labels     

    meta = {
        'topology': 'full_circle',
        'center_pre': mean.tolist(),  'scale_pre': scale,
        'radius_pre': radius,         'noise': noise,
        'geom_center_pre': geom_center_pre.tolist(),
        'center_post': center_post.tolist(),
        'radius_post': float(radius_post),
        'n_components': n
        }
    # print(Xn.shape, theta.shape) (n, 2), (n,))
    return {'X': Xn, 'params': theta.astype(np.float32), 'meta': meta}


def generate_ucg(n=10000, n_components=200, radius=8.0, noise=1.0):

    # ucg dataset from https://github.com/VictorBoutin/RiemannEBM/blob/main/tutorial/geodesics.ipynb

    rng = np.random.default_rng(0)
    angles_all = np.linspace(0, np.pi, n_components)
    means = radius * np.stack([np.cos(angles_all), np.sin(angles_all)], axis=-1)
    cluster_id = rng.integers(0, n_components, size=n)
    theta = angles_all[cluster_id]
    X = means[cluster_id] + noise * rng.normal(size=(n, 2))
    Xn, mean, scale = _normalize(X)
    geom_center_pre = np.zeros(2)
    center_post = (geom_center_pre - mean) / scale
    radius_post = radius / scale

    meta = {
        'topology': 'semicircle', 'theta_range': [0.0, float(np.pi)],
        'center_pre': mean.tolist(),  'scale_pre': scale,
        'radius_pre': radius,         'n_components': n_components, 'noise': noise,
        'geom_center_pre': geom_center_pre.tolist(),
        'center_post': center_post.tolist(),
        'radius_post': float(radius_post),
    }
    return {'X': Xn, 'params': theta.astype(np.float32), 'meta': meta}
 
def generate_wcg(n=10000, n_components=200, radius=8.0, noise=1.0):

    # wcg dataset from https://github.com/VictorBoutin/RiemannEBM/blob/main/tutorial/geodesics.ipynb

    rng = np.random.default_rng(0)
    angles_all = np.linspace(0, np.pi, n_components)
    means = radius * np.stack([np.cos(angles_all), np.sin(angles_all)], axis=-1)
    lin = np.concatenate([np.linspace(1, 30, 90), np.ones(11) * 30])
    weights = np.concatenate([lin, lin[1:-1][::-1]]); weights /= weights.sum()
    cluster_id = rng.choice(n_components, size=n, p=weights)
    theta = angles_all[cluster_id]
    X = means[cluster_id] + noise * rng.normal(size=(n, 2))
    Xn, mean, scale = _normalize(X)
    geom_center_pre = np.zeros(2)
    center_post = (geom_center_pre - mean) / scale
    radius_post = radius / scale

    meta = {
        'topology': 'semicircle_weighted', 'theta_range': [0.0, float(np.pi)],
        'center_pre': mean.tolist(),  'scale_pre': scale,
        'radius_pre': radius,         'n_components': n_components, 'noise': noise,
        'geom_center_pre': geom_center_pre.tolist(),
        'center_post': center_post.tolist(),
        'radius_post': float(radius_post),
    }
    return {'X': Xn, 'params': theta.astype(np.float32), 'meta': meta}

def generate_swiss_roll(n=10000, noise=0.2):
    X3, t = make_swiss_roll(n_samples=n, noise=noise, random_state=0)
    X = X3[:, [0, 2]]                                          # (n, 2)
    Xn, mean, scale = _normalize(X)
    meta = {
        'topology': 'spiral',
        'center_pre': mean.tolist(), 'scale_pre': scale, 'noise': noise,
        't_range': [float(t.min()), float(t.max())]
    }
    labels = ((t - t.min()) / (t.max() - t.min()) * 10).astype(np.int64)
    return {'X': Xn, 'labels': labels,
            'params': t.astype(np.float32), 'meta': meta}

def generate_two_moons(n=10000, noise=0.05):
    X, y = make_moons(n_samples=n, noise=noise, random_state=0)
    Xn, mean, scale = _normalize(X)
    meta = {'topology': 'two_moons', 'gt_available': False,
            'center_pre': mean.tolist(), 'scale_pre': scale, 'noise': noise}
    return {'X': Xn, 'labels': y.astype(np.int64),
            'params': np.zeros(n, dtype=np.float32), 'meta': meta}

def generate_s_curve(n=10000, noise=0.05):
    X3, t = make_s_curve(n_samples=n, noise=noise, random_state=0)
    X = X3[:, [2, 0]]   
    Xn, mean, scale = _normalize(X)
    meta = {
        'topology': 's-curve',
        'center_pre': mean.tolist(), 'scale_pre': scale, 'noise': noise,
        't_range': [float(t.min()), float(t.max())]
    }
    labels = ((t - t.min()) / (t.max() - t.min()) * 10).astype(np.int64)
    return {'X': Xn, 'labels': labels,
            'params': t.astype(np.float32), 'meta': meta}

datasets = {
    '1-sphere': generate_circle,   
    'ucg': generate_ucg,
    's-curve': generate_s_curve,
    'wcg': generate_wcg,
    'spiral': generate_swiss_roll,
    'two_moons': generate_two_moons,
}

def generate_dataset(name, data_root, **generate_kwargs):
    # generates dataset + saves to {data_root}/{name}/{data,params}.npy + meta.json
    if name not in datasets: 
        raise KeyError(f'Unkown dataset {name}. Known: {list(datasets)}')
    
    ds_dir = Path(data_root) / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    result = datasets[name](**generate_kwargs)
    np.save(ds_dir / 'data.npy', result['X'])
    np.save(ds_dir / 'params.npy', result['params'])
    with (ds_dir / 'meta.json').open('w') as f:
        json.dump(result['meta'], f, indent=2)
    return result, ds_dir

def load_dataset(name, data_root):
    # loads the dataset from npy files
    ds_dir = Path(data_root) / name 
    X = np.load(ds_dir / 'data.npy')
    params = np.load(ds_dir / 'params.npy')
    with (ds_dir / 'meta.json').open() as f:
        meta = json.load(f)
    return {'X': X, 'params': params, 'meta': meta}
