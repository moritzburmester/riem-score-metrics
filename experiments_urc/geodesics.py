import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent))

ckpt_diff = "/home/moritz.burmester/riem-score-metrics/experiments_urc/model_logs/alphanum/vp_stdw_big/checkpoints/best/epoch=4559--eval_loss_epoch=0.023.ckpt"
rae_ckpt  = "/home/moritz.burmester/riem-score-metrics/experiments_urc/rae_triplet.pt"
stats_path = "/home/moritz.burmester/riem-score-metrics/experiments_urc/latents_stats.npz"
real_lat  = "/home/moritz.burmester/riem-score-metrics/experiments_urc/latents.npy"

beta_min, beta_max = 0.1, 20.0
hidden_layers, hidden_nodes = 6, 512
n_angles = 180
deg_per_idx = 2.0
device = "cuda" if torch.cuda.is_available() else "cpu"

t_val = 0.05
lams = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
init_mode = "slerp"
n_geodesics = 100
gap_min_deg, gap_max_deg = 10.0, 180.0
n_points = 50
n_iter = 3000
lr, lr_min = 1e-3, 1e-4
n_strip = 3
seed = 0
outdir = here / "geo_out"


def lerp(z0, z1, u):
    return (1 - u) * z0 + u * z1


def slerp(z0, z1, u):
    n0 = z0 / z0.norm(dim=1, keepdim=True)
    n1 = z1 / z1.norm(dim=1, keepdim=True)
    dot = (n0 * n1).sum(1).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(dot); so = torch.sin(omega)
    a = (torch.sin((1 - u) * omega) / so)[:, None]
    b = (torch.sin(u * omega) / so)[:, None]
    res = a * z0 + b * z1
    return torch.where(so[:, None] > 1e-6, res, lerp(z0, z1, u))


def init_path(z0, z1, n, mode):
    f = slerp if mode == "slerp" else lerp
    us = torch.linspace(0, 1, n, device=z0.device)
    return torch.stack([f(z0, z1, float(u)) for u in us], dim=1)


def discrete_geodesic_score(score_fn, z0, z1, n_points, n_iter, lr, lr_min, lam, init):
    p, d = z0.shape
    n = n_points
    coef = 0.5 * (n - 1)

    z_init = init_path(z0, z1, n, init)
    z_i = z_init[:, 1:-1].clone().detach().requires_grad_(True)

    with torch.no_grad():
        s0 = score_fn(z0)[:, None, :]
        s1 = score_fn(z1)[:, None, :]

    def path_scores(zint):
        s_int = score_fn(zint.reshape(-1, d)).reshape(p, n - 2, d)
        return torch.cat([s0, s_int, s1], dim=1)

    if lam != 0.0:
        with torch.no_grad():
            s_all = path_scores(z_i)
            ds0 = s_all[:, 1:] - s_all[:, :-1]
            v0 = z_init[:, 1:] - z_init[:, :-1]
            w0 = s_all[:, :-1].pow(2).sum(-1)
            t_dir = ds0.pow(2).sum(-1).sum(1).clamp_min(1e-12)
            t_norm = (w0 * v0.pow(2).sum(-1)).sum(1).clamp_min(1e-12)
    else:
        t_dir = torch.ones(p, device=z0.device)
        t_norm = torch.ones(p, device=z0.device)

    opt = torch.optim.Adam([z_i], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr_min)

    for ep in range(n_iter):
        opt.zero_grad()
        s = path_scores(z_i)
        x = torch.cat([z0[:, None], z_i, z1[:, None]], dim=1)
        ds = s[:, 1:] - s[:, :-1]
        e_dir = ds.pow(2).sum(-1)
        if lam != 0.0:
            v = x[:, 1:] - x[:, :-1]
            w = s[:, :-1].pow(2).sum(-1)
            e_norm = w * v.pow(2).sum(-1)
            per_path = (1 - lam) * e_dir.sum(1) / t_dir + lam * e_norm.sum(1) / t_norm
        else:
            per_path = e_dir.sum(1) / t_dir
        loss = coef * per_path.sum()
        loss.backward()
        opt.step()
        sched.step()
        if ep == 0 or (ep + 1) % 500 == 0 or ep == n_iter - 1:
            print(f"   ep={ep+1:>4d} E={loss.item():.3e}")

    with torch.no_grad():
        paths = torch.cat([z0[:, None], z_i, z1[:, None]], dim=1).detach().cpu()
    return paths


def apply_ema_(pl, path):
    ckpt = torch.load(path, map_location="cpu")
    ema = (ckpt.get("optimizer_states") or [{}])[0].get("ema", None)
    params = list(pl.score_model.parameters())
    if ema is None or len(ema) != len(params):
        print("  [ema] not applied -> raw weights")
        return
    with torch.no_grad():
        for q, e in zip(params, ema):
            q.data.copy_(e.to(q.device))
    print(f"  [ema] applied {len(ema)} tensors")


def load_score_fn(t_val):
    sys.path.insert(0, '/home/moritz.burmester/riem-score-metrics/diffusion_model_dependencies')
    from lightning_modules import BaseSdeGenerativeModel
    from models import fcn
    import importlib
    importlib.import_module("lightning_data_modules.Toy2DDataset")
    from lightning_modules.utils import create_lightning_module
    from models.utils import get_score_fn
    from experiments_toy_datasets.model_utils import build_config

    cfg = build_config("alphanum", ambient_dim=64, sde="vpsde",
                       beta_min=beta_min, beta_max=beta_max,
                       hidden_layers=hidden_layers, hidden_nodes=hidden_nodes,
                       epochs=6000, batch_size=128, lr=2e-4)
    pl = create_lightning_module(cfg)
    pl = type(pl).load_from_checkpoint(ckpt_diff)
    pl.config = cfg
    pl.configure_sde(cfg)
    pl = pl.to(device).eval()
    apply_ema_(pl, ckpt_diff)
    sde = pl.sde

    with torch.no_grad():
        sig = sde.marginal_prob(torch.zeros(1, 64, device=device),
                                torch.tensor([t_val], device=device))[1].item()
    print(f"  [metric] t={t_val} -> sigma(t)={sig:.4f} (vp)")

    raw = get_score_fn(sde, pl.score_model, conditional=False, train=False, continuous=True)

    def score_fn(x):
        return raw(x, torch.full((x.shape[0],), float(t_val), device=x.device))
    return score_fn


def load_rae():
    from rae import RAE2
    m = RAE2(in_ch=1, nb_feature=128, z_dim=64).to(device)
    m.load_state_dict(torch.load(rae_ckpt, map_location=device))
    m.eval()
    return m


def sample_endpoints(real, n, gap_min_deg, gap_max_deg, rng):
    letters = rng.integers(0, 7, size=n)
    a0 = rng.integers(0, n_angles, size=n)
    gmin = int(round(gap_min_deg / deg_per_idx))
    gmax = int(round(gap_max_deg / deg_per_idx))
    steps = rng.integers(gmin, gmax + 1, size=n)
    a1 = (a0 + steps) % n_angles
    z0 = real[letters * n_angles + a0]
    z1 = real[letters * n_angles + a1]
    return (torch.tensor(z0, dtype=torch.float32, device=device),
            torch.tensor(z1, dtype=torch.float32, device=device),
            letters, a0, steps)


def decode_strip(rae, stats, path, out_png, title=None):
    mean, std = float(stats["mean"]), float(stats["std"])
    z = torch.tensor(path.numpy() * std + mean, dtype=torch.float32, device=device)
    with torch.no_grad():
        imgs = ((rae.decode(z).clamp(-1, 1) + 1) / 2).cpu().numpy()
    n = imgs.shape[0]
    fig, ax = plt.subplots(1, n, figsize=(n, 1.3))
    for k in range(n):
        ax[k].imshow(imgs[k, 0], cmap="gray", vmin=0, vmax=1)
        ax[k].set_xticks([]); ax[k].set_yticks([])
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    score_fn = load_score_fn(t_val)
    rae = load_rae()
    stats = np.load(stats_path)
    real = np.load(real_lat).astype(np.float32)

    z0, z1, letters, a0, steps = sample_endpoints(real, n_geodesics,
                                                  gap_min_deg, gap_max_deg, rng)
    np.savez(outdir / "endpoints.npz", letters=letters, a0=a0, steps=steps,
             z0=z0.cpu().numpy(), z1=z1.cpu().numpy())
    print(f"{n_geodesics} geodesics, gap {gap_min_deg}-{gap_max_deg}deg, "
          f"N={n_points}, t={t_val}, init={init_mode}")

    for lam in lams:
        tag = f"{init_mode}_lam{lam}".replace(".", "")
        print(f"optimizing [{tag}] ...")
        paths = discrete_geodesic_score(score_fn, z0, z1, n_points, n_iter,
                                        lr, lr_min, lam, init_mode)
        torch.save({"paths": paths, "letters": letters, "a0": a0, "steps": steps,
                    "t": t_val, "lam": lam, "init": init_mode}, outdir / f"geo_{tag}.pt")
        for p in range(min(n_strip, paths.shape[0])):
            decode_strip(rae, stats, paths[p], outdir / f"strip_{tag}_g{p}.png",
                         title=f"{tag} g{p} (L{letters[p]} "
                               f"{a0[p]*deg_per_idx:.0f}->{((a0[p]+steps[p])%n_angles)*deg_per_idx:.0f}deg)")
    print(f"done -> {outdir}/")