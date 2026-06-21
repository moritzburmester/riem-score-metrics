from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import lightning_data_modules.utils as utils


## Code adapted from https://github.com/GBATZOLIS/ID-diff/blob/main/lightning_data_modules/ImageDatasets.py 
## and https://github.com/yang-song/score_sde_pytorch


class Toy2DDataset(Dataset):
    # also works for any D (e.g. rotated chars latent space)
    def __init__(self, config):
        super().__init__()
        base = Path(config.data.base_dir)
        fname = config.data.get('latents_file', 'data.npy')
        X = np.load(base / fname).astype(np.float32)
        if X.ndim != 2:
            raise ValueError(f'expected 2D array (N, D); got shape {X.shape}')
        ad = config.data.get('ambient_dim', X.shape[1])
        if ad != X.shape[1]:
            raise ValueError(
                f'config.data.ambient_dim = {ad} but the .npy has D = {X.shape[1]}'
            )
        self.data = torch.from_numpy(X)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


@utils.register_lightning_datamodule(name='Toy2D')
class Toy2DDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.split = config.data.split

        self.train_workers = config.training.workers
        self.val_workers   = config.validation.workers
        self.test_workers  = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch   = config.validation.batch_size
        self.test_batch  = config.eval.batch_size

    def setup(self, stage=None):
        self.dataset = Toy2DDataset(self.config)
        n = len(self.dataset)
        tr = int(self.split[0] * n)
        va = int(self.split[1] * n)
        te = n - tr - va
        self.train_data, self.valid_data, self.test_data = random_split(self.dataset, [tr, va, te], generator=torch.Generator().manual_seed(self.config.seed))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch,
                          num_workers=self.train_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.val_batch,
                          num_workers=self.val_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.test_batch,
                          num_workers=self.test_workers, shuffle=False)
