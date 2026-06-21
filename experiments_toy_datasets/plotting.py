import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from .toy_datasets_utils import load_dataset
from .manifolds.score_analytic import get_gmm_components, _gmm_log_density, gmm_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "experiments_toy_datasets" / "toy_data"

POINT_SIZE = 2.0
POINT_ALPHA = 0.35
LEVELS = 30
ALPHA_BG = 0.85
DENSITY_CMAP = "Blues_r"
SCORE_CMAP = "Purples_r"

# Apply the house style once, on import.
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
})


def _limits(X, pad=0.1):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_pad = (x_max - x_min) * pad
    y_pad = (y_max - y_min) * pad
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _to_np(a):
    if torch.is_tensor(a):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _new_ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))
    return ax


def _style_ax(ax, name, xlim, ylim):
    ax.set_facecolor("#fafafa")
    ax.grid(alpha=0.15)
    ax.set_title(name, fontsize=10, pad=8, weight="semibold")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


def plot_dataset(name, ax=None, data_dir=DATA_DIR, pad=0.1):
    X = load_dataset(name, data_dir)["X"]
    xlim, ylim = _limits(X, pad)

    ax = _new_ax(ax)
    ax.scatter(
        X[:, 0],
        X[:, 1],
        s=POINT_SIZE,
        color="black",
        alpha=POINT_ALPHA,
        edgecolors="none",
        zorder=2,
        rasterized=True,
    )
    _style_ax(ax, name, xlim, ylim)
    return ax


def plot_density(name, ax=None, data_dir=DATA_DIR, pad=0.1,
                 grid_size=200, n_components=500):
    dataset = load_dataset(name, data_dir)
    X, meta = dataset["X"], dataset["meta"]
    xlim, ylim = _limits(X, pad)

    means, sigma, weights = get_gmm_components(name, meta, n_components=n_components)
    means = means.numpy()

    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    logp = _gmm_log_density(
        grid, means, sigma,
        weights.numpy() if weights is not None else None,
    ).reshape(grid_size, grid_size)

    ax = _new_ax(ax)
    ax.contourf(xx, yy, logp, levels=LEVELS, cmap=DENSITY_CMAP,
                alpha=ALPHA_BG, zorder=0)
    _style_ax(ax, name, xlim, ylim)
    return ax


def plot_score_magnitude(name, ax=None, data_dir=DATA_DIR, pad=0.1,
                         grid_size=200, n_components=500):
    dataset = load_dataset(name, data_dir)
    X, meta = dataset["X"], dataset["meta"]
    xlim, ylim = _limits(X, pad)

    means, sigma, weights = get_gmm_components(name, meta, n_components=n_components)

    x = torch.linspace(xlim[0], xlim[1], grid_size)
    y = torch.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)

    s = gmm_score(grid, means, sigma, weights=weights)
    s_mag = torch.norm(s, dim=1).reshape(grid_size, grid_size)

    ax = _new_ax(ax)
    ax.contourf(xx.numpy(), yy.numpy(), s_mag.numpy(), levels=LEVELS,
                cmap=SCORE_CMAP, alpha=ALPHA_BG, zorder=0)
    _style_ax(ax, name, xlim, ylim)
    return ax


def plot_curve(name, curve_path, ax=None, data_dir=DATA_DIR, pad=0.1,
               background="density", show_data=True, grid_size=200,
               n_components=500, lw=2.0):
    paths = [curve_path] if isinstance(curve_path, (str, Path)) else list(curve_path)

    X = load_dataset(name, data_dir)["X"]
    xlim, ylim = _limits(X, pad)

    ax = _new_ax(ax)

    # background field (reuses the contour plotters)
    if background == "density":
        plot_density(name, ax=ax, data_dir=data_dir, pad=pad,
                     grid_size=grid_size, n_components=n_components)
    elif background == "score":
        plot_score_magnitude(name, ax=ax, data_dir=data_dir, pad=pad,
                             grid_size=grid_size, n_components=n_components)

    if show_data:
        ax.scatter(X[:, 0], X[:, 1], s=POINT_SIZE, color="black",
                   alpha=POINT_ALPHA, edgecolors="none", zorder=2,
                   rasterized=True)

    cmap = plt.get_cmap("tab10")
    multi = len(paths) > 1
    endpoints_drawn = False
    for i, p in enumerate(paths):
        rec = torch.load(p, map_location="cpu")
        curve = _to_np(rec["curve"])
        color = cmap(i % 10) if multi else "#c1121f"
        label = rec.get("metric") if multi else None
        ax.plot(curve[:, 0], curve[:, 1], color=color, lw=lw, zorder=3,
                label=label)
        if not endpoints_drawn:
            z0, z1 = _to_np(rec["z0"]), _to_np(rec["z1"])
            ax.scatter([z0[0], z1[0]], [z0[1], z1[1]], color="black", s=45,
                       zorder=4, edgecolors="white", linewidths=1.0)
            endpoints_drawn = True

    if multi:
        ax.legend(loc="best", fontsize=8, framealpha=0.7)

    _style_ax(ax, name, xlim, ylim)
    return ax


_PLOTTERS = {
    "data": plot_dataset,
    "density": plot_density,
    "score": plot_score_magnitude,
}


def plot_grid(names, kind="data", data_dir=DATA_DIR, subplot_size=6.5, **kwargs):

    plot_fn = _PLOTTERS[kind]

    n_cols = int(np.ceil(np.sqrt(len(names))))
    n_rows = int(np.ceil(len(names) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(subplot_size * n_cols, subplot_size * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    plot_idx = 0
    for name in names:
        try:
            plot_fn(name, ax=axes[plot_idx], data_dir=data_dir, **kwargs)
        except FileNotFoundError:
            print(f"{name}: not generated yet")
            continue
        plot_idx += 1

    for ax in axes[plot_idx:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes

def plot_datasets(names, data_dir=DATA_DIR, subplot_size=6.5):
    return plot_grid(names, kind="data", data_dir=data_dir, subplot_size=subplot_size)