import sys
import time
from pathlib import Path
from datetime import timedelta
import ml_collections
import torch

sys.path.insert(0, '/home/moritz.burmester/riem-score-metrics/diffusion_model_dependencies')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_dir = Path('/home/moritz.burmester/riem-score-metrics/experiments_toy_datasets/model_logs')
data_dir = Path('/home/moritz.burmester/riem-score-metrics/experiments_rotatedcharacters')

# will be overwritten 
sde_defaults = dict(sde='vesde', sigma_min=0.01, sigma_max=5.0)
train_defaults = dict(epochs=1000, batch_size=128, lr=1e-3,
                      hidden_layers=4, hidden_nodes=128)


def build_config(name, **overrides):
    from configs.default import get_default_configs

    sde_p = {**sde_defaults, **{k: v for k, v in overrides.items() if k in sde_defaults}}
    train_p = {**train_defaults, **{k: v for k, v in overrides.items() if k in train_defaults}}
    D = overrides.get('ambient_dim', 2)
    g = overrides.get

    cfg = get_default_configs()
    cfg.logging.log_path = str(models_dir)
    cfg.logging.log_name = name
    cfg.logging.top_k = 1
    cfg.logging.every_n_epochs = 50
    cfg.logging.envery_timedelta = timedelta(minutes=5)
    cfg.logging.svd_frequency = 50
    cfg.logging.save_svd = False
    cfg.logging.svd_points = 25

    cfg.training.mode = 'train'
    cfg.training.gpus = 1 if torch.cuda.is_available() else 0
    cfg.training.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    cfg.training.lightning_module = 'base'
    cfg.training.batch_size = train_p['batch_size']
    cfg.training.num_epochs = train_p['epochs']
    cfg.training.n_iters = int(1e20)
    cfg.training.likelihood_weighting = bool(g('likelihood_weighting', True))
    cfg.training.reduce_mean = bool(g('reduce_mean', False))
    cfg.training.continuous = True
    cfg.training.sde = sde_p['sde']
    cfg.training.visualization_callback = []
    cfg.training.show_evolution = False
    cfg.training.snapshot_sampling = bool(g('snapshot_sampling', True))
    cfg.training.workers = int(g('workers', cfg.training.workers))

    cfg.validation.batch_size = 500
    cfg.validation.workers = int(g('workers', cfg.validation.workers))
    cfg.sampling.method = 'pc'
    cfg.sampling.predictor = 'reverse_diffusion'
    cfg.sampling.corrector = 'none'
    cfg.sampling.n_steps_each = 1
    cfg.sampling.noise_removal = True
    cfg.sampling.probability_flow = False
    cfg.sampling.snr = 0.25

    cfg.data = ml_collections.ConfigDict()
    cfg.data.datamodule = 'Toy2D'
    cfg.data.base_dir = str(data_dir / name)
    cfg.data.latents_file = 'data.npy'
    cfg.data.create_dataset = False
    cfg.data.split = [0.8, 0.1, 0.1]
    cfg.data.data_samples = 10_000
    cfg.data.use_data_mean = False
    cfg.data.ambient_dim = D
    cfg.data.dim = D
    cfg.data.num_channels = 0
    cfg.data.shape = [D]

    cfg.model = ml_collections.ConfigDict()
    cfg.model.checkpoint_path = None
    cfg.model.sigma_min = sde_p['sigma_min']
    cfg.model.sigma_max = sde_p['sigma_max']
    cfg.model.beta_min = float(g('beta_min', 0.1))
    cfg.model.beta_max = float(g('beta_max', 20.0))
    cfg.model.name = 'fcn'
    cfg.model.state_size = D
    cfg.model.hidden_layers = train_p['hidden_layers']
    cfg.model.hidden_nodes = train_p['hidden_nodes']
    cfg.model.dropout = 0.0
    cfg.model.scale_by_sigma = False
    cfg.model.num_scales = 1000
    cfg.model.ema_rate = float(g('ema_rate', 0.999))

    cfg.optim.weight_decay = float(g('weight_decay', 0.0))
    cfg.optim.optimizer = 'Adam'
    cfg.optim.lr = train_p['lr']
    cfg.optim.beta1 = 0.9
    cfg.optim.eps = 1e-8
    cfg.optim.warmup = int(g('warmup', 200))
    cfg.optim.grad_clip = 1.0

    cfg.seed = int(g('seed', 12))
    cfg.device = device
    cfg.dim_estimation = ml_collections.ConfigDict()
    return cfg


if __name__ == '__main__':

    name = 'alphanum'
    tag = 'vp_stdw_big'

    epochs = 6000
    ds_dir = Path('/home/moritz.burmester/riemannian-score-metrics/experiments_rotatedcharacters')

    overrides = dict(
        sde='vpsde',
        likelihood_weighting=False,
        reduce_mean=True,
        hidden_layers=6,
        hidden_nodes=512,
        epochs=epochs,
        ambient_dim=64,
        run_tag=tag,
        workers=0,
        snapshot_sampling=False,
    )

    from lightning_modules import BaseSdeGenerativeModel
    from models import fcn
    import importlib
    importlib.import_module('lightning_data_modules.Toy2DDataset')
    import run_lib

    cfg = build_config(name, **overrides)
    cfg.logging.log_name = f'{name}/{tag}'
    cfg.data.base_dir = str(ds_dir)
    cfg.data.latents_file = 'latents.npy'

    rd = models_dir / name / tag
    rd.mkdir(parents=True, exist_ok=True)

    print(f'[{tag}] sde={cfg.training.sde} lw={cfg.training.likelihood_weighting} '
          f'rm={cfg.training.reduce_mean} arch=fcn({cfg.model.hidden_layers}x{cfg.model.hidden_nodes}) '
          f'epochs={cfg.training.num_epochs}', flush=True)

    run_lib.train(cfg, log_path=cfg.logging.log_path,
                  checkpoint_path=None, log_name=cfg.logging.log_name)
