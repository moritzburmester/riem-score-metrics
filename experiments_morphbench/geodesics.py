import sys
import argparse
from pathlib import Path
from PIL import Image
import torch

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))

from helpers import *
from sd_wrapper import load_wrapper
from text_inversion import TextInversion
from lerp_slerp import lerp, slerp

device = 'cuda'

mbm_prompts = {
    'anakin': ('a photo of a young man with blue eyes', 'a photo of darth vader from star wars'),
    'arya': ('a photo of a woman holding a sword', 'a photo of a girl with long hair and a sword'),
    'boy01': ('a photo of a young boy in a hoodie', 'a photo of a young boy with curly hair'),
    'boy02': ('a photo of a young boy in a hoodie', 'a photo of a young boy with blonde hair'),
    'boy03': ('a photo of a young boy in a hoodie', 'a photo of a young boy in the rain'),
    'boy12': ('a photo of a young boy with curly hair', 'a photo of a young boy with blonde hair'),
    'boy13': ('a photo of a young boy with curly hair', 'a photo of a young boy in the rain'),
    'boy23': ('a photo of a young boy with blonde hair', 'a photo of a young boy in the rain'),
    'boy_girl_0': ('a photo of a young boy looking at the camera', 'a photo of a young girl with blue eyes'),
    'boy_girl_1': ('a photo of a young boy with curly hair', 'a photo of a young woman with green eyes'),
    'boy_girl_2': ('a photo of a young boy with blonde hair', 'a photo of a young girl with blonde hair'),
    'boy_girl_3': ('a photo of a young boy in the rain', 'a photo of a young girl with her hair blowing in the wind'),
    'cake_burger': ('a photo of a stack of pancakes with berries on top', 'a photo of a hamburger on a black plate'),
    'carr_cars': ('a photo of a red maserati convertible parked on a pier', 'a photo of the mercedes amg gt3 race car'),
    'cars_van': ('a photo of the mercedes amg gt3 race car', 'a photo of a silver mini cooper on a gray floor'),
    'castle': ('a photo of mont saint michel in normandy, france', 'a photo of mont saint michel at dusk'),
    'cat_rabbit': ('a photo of an orange and white cat looking up at the window', 'a photo of a rabbit sitting on the grass'),
    'chair01': ('a photo of a blue chair sitting on a hardwood floor', 'a photo of a green couch in front of a white wall'),
    'chair02': ('a photo of a blue chair sitting on a hardwood floor', 'a photo of an office chair with wooden legs'),
    'chair03': ('a photo of a blue chair sitting on a hardwood floor', 'a photo of a white chair next to a table'),
    'chair12': ('a photo of a green couch in front of a white wall', 'a photo of an office chair with wooden legs'),
    'chair13': ('a photo of a green couch in front of a white wall', 'a photo of a white chair next to a table'),
    'chair23': ('a photo of an office chair with wooden legs', 'a photo of a white chair next to a table'),
    'dog': ('a photo of a dog sitting in front of a brown background', 'a photo of a black and white dog sitting in the grass'),
    'dog_wolf': ('a photo of a corgi puppy sitting on an orange background', 'a photo of a gray wolf looking at the camera'),
    'gandalf': ('a photo of an old man with a pipe in his mouth', 'a photo of santa claus with a beard and glasses'),
    'girl': ('a photo of a woman in a black jacket', 'a photo of a young woman with brown hair'),
    'girl01': ('a photo of a young girl with blue eyes', 'a photo of a young woman with green eyes'),
    'girl02': ('a photo of a young girl with blue eyes', 'a photo of a young girl with blonde hair'),
    'girl03': ('a photo of a young girl with blue eyes', 'a photo of a young girl with her hair blowing in the wind'),
    'girl12': ('a photo of a young woman with green eyes', 'a photo of a young girl with blonde hair'),
    'girl13': ('a photo of a young woman with green eyes', 'a photo of a young girl with her hair blowing in the wind'),
    'girl23': ('a photo of a young girl with blonde hair', 'a photo of a young girl with her hair blowing in the wind'),
    'house_left': ('a photo of an old house in the middle of a field', 'a photo of a barn with a red roof'),
    'house_lr': ('a photo of an old house in the middle of a field', 'a photo of an old wooden cabin in the middle of a grassy field'),
    'house_right': ('a photo of an old wooden cabin in the middle of a grassy field', 'a photo of a red house covered in snow'),
    'jay': ('a photo of asian man with black hair', 'a photo of an asian man in a suit and tie'),
    'leo': ('a photo of a young man with short hair', 'a photo of a man in a suit and tie'),
    'lion_tiger': ('a photo of a lion resting on a rock', 'a photo of a tiger with its mouth open'),
    'man_van': ('a photo of a man wearing a black suit and a black hat', 'a photo of a painting of a man wearing a hat'),
    'mona_pearl': ('a photo of the mona lisa by leonardo da vinci', 'a photo of a girl with a pearl earring'),
    'Musk_Feifei': ('a photo of a man in a suit and tie', 'a photo of an asian woman in a blue shirt'),
    'Musk_Obama': ('a photo of a man in a suit and tie', 'a photo of president barack obama'),
    'Musk_Trump': ('a photo of a man in a suit and tie', 'a photo of a man wearing a suit and tie'),
    'obama_putin': ('a photo of president barack obama', 'a photo of russian president vladimir putin'),
    'Obama_Trump': ('a photo of president barack obama', 'a photo of donald trump'),
    'pika': ('a photo of a pikachu on a white background', 'a photo of a pikachu with a lightning bolt coming out of its mouth'),
    'raccoon': ('a photo of a raccoon looking at the camera', 'a photo of a raccoon with blue eyes'),
    'realdog_cat': ('a photo of a golden retriever puppy', 'a photo of an orange and white cat'),
    'red_car': ('a photo of a red maserati convertible parked on a pier', 'a photo of a small red car parked on the side of the road'),
    'scream': ('a photo of home alone 2 double pack', 'a photo of the scream by edvard munch'),
    'sculp': ('a photo of a marble head with curly hair', 'a photo of a bust of a man with curly hair'),
    'snow_mountain': ('a photo of a mountain range covered in snow', 'a photo of the milky way in the night sky over a mountain'),
    'taylor_yifei': ('a photo of taylor swift with red lipstick', 'a photo of an asian woman with long hair'),
    'thanos': ('a photo of thanos in fortnite', 'a photo of superman flying through the air'),
    'thu_mit': ('a photo of the entrance to a university building', 'a photo of people sitting on the grass in front of a large building'),
    'Trump_Biden': ('a photo of a man wearing a suit and tie', 'a photo of joe biden in front of the american flag'),
    'vangogh': ('a photo of a painting of vincent van gogh', 'a photo of a painting of vincent van gogh'),
    'van_jeep': ('a photo of a silver mini cooper on a gray floor', 'a photo of the mercedes g - class pickup truck'),
    'van_mona': ('a photo of a painting of a man wearing a hat', 'a photo of the mona lisa by leonardo da vinci'),
    'van_pearl': ('a photo of a painting of vincent van gogh', 'a photo of a painting of a girl with a pearl earring'),
    'van_self': ('a photo of a painting of vincent van gogh', 'a photo of a painting of a man wearing a hat'),
    'wave': ('a photo of a large wave in the ocean', 'a photo of the great wave off kanagawa'),
    'wc': ('a photo of a restroom sign with a man and woman', 'a photo of a man and woman dancing in a yellow dress'),
    'whitehouse_church': ('a photo of the U.S. Capitol building in Washington, DC', 'a photo of the cathedral in florence, italy'),
    'wolf_tiger': ('a photo of a gray wolf looking at the camera', 'a photo of a tiger with its mouth open'),
}

negative_prompt = ("A doubling image, unrealistic, artifacts, distortions, unnatural blending, "
                   "ghosting effects, overlapping edges, harsh transitions, motion blur, "
                   "poor resolution, low detail")


def slerp_init_path(z0, z1, n_points):
    us = torch.linspace(0, 1, n_points)
    pts = [slerp(z0.reshape(1, 4, 64, 64), z1.reshape(1, 4, 64, 64), float(u)).reshape(-1)
           for u in us]
    return torch.stack(pts, 0).to(device)


def make_sd_score_fn_neg(sd, neg_prompt, t_scalar):
    t_int = int(t_scalar)
    abar = sd.scheduler.alphas_cumprod[t_int].to(sd.device)
    sigma = (1.0 - abar).sqrt()
    emb_neg = sd.prompt2embed(neg_prompt).float()

    def score_fn(x, emb):
        m = x.shape[0]
        lat = x.reshape(m, 4, 64, 64).to(sd.device, torch.float32)
        t_batch = torch.full((m,), t_int, device=sd.device, dtype=torch.long)
        eps_c = sd.noise_pred(lat, t_batch, emb)
        eps_neg = sd.noise_pred(lat, t_batch, emb_neg.repeat(m, 1, 1))
        s = -(eps_c - eps_neg) / sigma
        return s.reshape(m, -1)

    return score_fn


def discrete_geodesic_score(score_fn, z0, z1, emb_cond, n_points=11, n_iter=500,
                            lr=1e-3, lr_min=1e-4, lam=0.0, init='slerp',
                            verbose=True, log_every=50, tag=''):
    n = n_points
    z0 = z0.to(device).flatten()
    z1 = z1.to(device).flatten()
    emb_cond = emb_cond.to(device)

    if init == 'slerp':
        z_init = slerp_init_path(z0, z1, n)
    else:
        t = torch.linspace(0, 1, n, device=device).unsqueeze(-1)
        z_init = z0[None] * (1 - t) + z1[None] * t

    z_i = z_init[1:-1].clone().detach().requires_grad_(True)

    with torch.no_grad():
        s_ends = score_fn(torch.stack([z0, z1], dim=0), emb_cond[[0, -1]])
    s0, s1 = s_ends[0:1], s_ends[1:2]

    if lam != 0.0:
        with torch.no_grad():
            s_all = torch.cat([s0, score_fn(z_init[1:-1], emb_cond[1:-1]), s1], dim=0)
            ds0 = s_all[1:] - s_all[:-1]
            v0 = z_init[1:] - z_init[:-1]
            t_dir = ds0.pow(2).sum().item()
            w0 = s_all[:-1].pow(2).sum(-1)
            t_norm = (w0 * v0.pow(2).sum(-1)).sum().item()
    else:
        t_dir = t_norm = 1.0

    opt = torch.optim.Adam([z_i], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr_min)
    du = 1.0 / (n - 1)
    coef = 0.5 / du

    history = []
    for ep in range(n_iter):
        opt.zero_grad()

        s_int = score_fn(z_i, emb_cond[1:-1])
        s = torch.cat([s0, s_int, s1], dim=0)
        x = torch.cat([z0[None], z_i, z1[None]], dim=0)

        ds = s[1:] - s[:-1]
        e_dir = ds.pow(2).sum(-1)

        if lam != 0.0:
            v = x[1:] - x[:-1]
            sn2 = s[:-1].pow(2).sum(-1)
            e_norm = sn2 * v.pow(2).sum(-1)
            loss = coef * ((1.0 - lam) * e_dir / t_dir + lam * e_norm / t_norm).sum()
        else:
            e_norm = None
            loss = coef * e_dir.sum()

        loss.backward()
        opt.step()
        sched.step()

        with torch.no_grad():
            direc = coef * e_dir.sum().item() / t_dir
            nrm = coef * e_norm.sum().item() / t_norm if e_norm is not None else 0.0
        history.append({'E': loss.item(), 'dir': direc, 'norm': nrm,
                        't_dir': t_dir, 't_norm': t_norm})

        if verbose and (ep == 0 or (ep + 1) % log_every == 0 or ep == n_iter - 1):
            lr_now = sched.get_last_lr()[0]
            print(f'      [{tag} lam={lam}] ep={ep+1:>4d} e={loss.item():.3e} '
                  f'dir={direc:.3e} norm={nrm:.3e} lr={lr_now:.1e}')

    with torch.no_grad():
        z_path = torch.cat([z0[None], z_i, z1[None]], dim=0)
    return z_path.detach().cpu(), history


def perturb_latent(z, eps, seed):
    g = torch.Generator(device=z.device).manual_seed(seed)
    n = torch.randn(z.shape, generator=g, device=z.device, dtype=z.dtype)
    return z + eps * z.pow(2).mean().sqrt() * n


def setup(sd, suffix, ti_key, prompt_a, prompt_b, img_a, img_b, n, t_level,
          eps=0.0, seed=15):
    f = here / "ckpt" / f"{suffix}_setup.pt"
    if f.exists():
        d = torch.load(f)
        print("loaded latents/embeddings")
        return d["zt_a"], d["zt_b"], d["emb_cond"]

    z_a = sd.img2latent(img_a)
    z_b = sd.img2latent(img_b)

    ti = TextInversion(sd, tv_lr=0.005, tv_steps=500, tv_batch_size=1,
                       tv_ckpt_folder=str(here / "ckpt"))
    emb_a = ti.text_inversion_load(prompt_a, z_a, prefix=f"mbm_{ti_key}_A").float()
    emb_b = ti.text_inversion_load(prompt_b, z_b, prefix=f"mbm_{ti_key}_B").float()

    if eps > 0:
        z_a = perturb_latent(z_a, eps, seed)
        out = here / 'geodesics'
        out.mkdir(exist_ok=True)
        sd.latent2img(z_a).save(out / f"{suffix}_perturbed_endpoint.png")

    us = torch.linspace(0, 1, n).tolist()
    emb_cond = lerp_cond_embed(us, emb_a, emb_b)

    zt_a = sd.latent_forward_inversion(z_a, emb_a, noise_level=t_level)
    zt_b = sd.latent_forward_inversion(z_b, emb_b, noise_level=t_level)

    f.parent.mkdir(exist_ok=True)
    torch.save({"zt_a": zt_a, "zt_b": zt_b, "emb_cond": emb_cond,
                "eps": eps, "seed": seed, "t_level": t_level}, f)
    print("saved latents/embeddings")
    return zt_a, zt_b, emb_cond


def denoise_and_save_images(sd, pts, emb_cond, tag, t_level):
    imgs = []
    for i in range(pts.shape[0]):
        zt = pts[i].reshape(1, 4, 64, 64).to(device)
        emb_i = emb_cond[i:i+1].to(device)
        imgs.append(sd.latent2img(sd.latent_backward(zt, emb_i, noise_level=t_level)))
    out = here / 'geodesics'
    out.mkdir(exist_ok=True)
    display_alongside(imgs).save(out / f"{tag}_strip.png")
    print(f"saved images for {tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', type=str, nargs='+', default=None)
    ap.add_argument('--lams', type=float, nargs='+', default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--z0noise', action='store_true')
    ap.add_argument('--eps', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=15)
    ap.add_argument('--t_level', type=float, default=0.6)
    ap.add_argument('--init', type=str, default='slerp', choices=['lerp', 'slerp'])
    ap.add_argument('--num_runs', type=int, default=32)
    args = ap.parse_args()

    if args.names is None:
        args.names = (here / 'subset32.txt').read_text().split()

    torch.manual_seed(args.seed)
    sd = load_wrapper()
    sd.unet.enable_gradient_checkpointing()

    mbm_dir = Path('/home/moritz.burmester/riemannian-score-metrics/'
                   'image_data/MorphBench/Metamorphosis')
    n = 11
    t_scalar = int(sd.get_t(args.t_level, return_single=True))
    score_fn = make_sd_score_fn_neg(sd, negative_prompt, t_scalar)

    runs = 0
    for name in args.names:
        if runs >= args.num_runs:
            break
        prompt_a, prompt_b = mbm_prompts[name]
        img_a = Image.open(mbm_dir / f"{name}_0.png")
        img_b = Image.open(mbm_dir / f"{name}_1.png")
        print(f"mbm pair: {name} | a={prompt_a!r} b={prompt_b!r}")

        if args.z0noise:
            suffix = f"{name}_z0noise{args.eps}_init{args.init}_t{args.t_level}".replace('.', '')
            ti_key = name
            eps = args.eps
        else:
            suffix = f"{name}_init{args.init}_t{args.t_level}".replace('.', '')
            ti_key = name
            eps = 0.0

        zt_a, zt_b, emb_cond = setup(sd, suffix, ti_key, prompt_a, prompt_b,
                                     img_a, img_b, n, args.t_level,
                                     eps=eps, seed=args.seed)

        us = torch.linspace(0, 1, n)
        for method, fn in [('lerp', lerp), ('slerp', slerp)]:
            tag = f"mbm_{suffix}_{method}"
            f = here / "ckpt" / f"geodesic_{tag}.pt"
            if f.exists():
                continue
            pts = torch.stack([fn(zt_a, zt_b, float(u)).reshape(-1) for u in us], dim=0).cpu()
            torch.save({'pts': pts, 'name': name, 'method': method,
                        'z0noise': args.z0noise, 'eps': args.eps if args.z0noise else None,
                        'seed': args.seed, 't_level': args.t_level, 'init': args.init}, f)
            denoise_and_save_images(sd, pts, emb_cond, tag, args.t_level)

        for lam in args.lams:
            tag = f"mbm_{suffix}_lam{lam}_it{args.iters}".replace('.', '')
            f = here / "ckpt" / f"geodesic_{tag}.pt"
            if f.exists():
                continue
            print(f"optimizing geodesic for {name}, lam={lam}")
            sd.unet.train()
            pts, history = discrete_geodesic_score(
                score_fn, zt_a.reshape(-1), zt_b.reshape(-1), emb_cond,
                lam=lam, n_points=n, n_iter=args.iters, lr=1e-3, lr_min=1e-4,
                init=args.init, verbose=True, tag=name)
            sd.unet.eval()
            print(f"peak vram: {torch.cuda.max_memory_allocated()/1e9:.2f} gb")

            torch.save({'pts': pts, 'history': history, 'lam': lam, 'name': name,
                        'z0noise': args.z0noise, 'eps': args.eps if args.z0noise else None,
                        'seed': args.seed, 'iters': args.iters,
                        't_level': args.t_level, 'init': args.init,
                        't_dir': history[-1]['t_dir'], 't_norm': history[-1]['t_norm']}, f)
            denoise_and_save_images(sd, pts, emb_cond, tag, args.t_level)

        runs += 1