import torch.nn as nn
import torch.nn.functional as F
import torch


## code from https://github.com/VictorBoutin/RiemannEBM/blob/main/model/rae.py

class RaeBlock(nn.Module):
    def __init__(self, dim_in, dim_out, ds, padding=1, use_bn=True, output_padding=0):
        super().__init__()
        self.use_bn = use_bn

        if ds:
            self.block = nn.Sequential(
                nn.BatchNorm2d(dim_in) if self.use_bn else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=4,
                    stride=2,
                    padding=padding,
                    bias=not self.use_bn,
                ))
        else:
            self.block = nn.Sequential(
                nn.BatchNorm2d(dim_in) if self.use_bn else nn.Identity(),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    dim_in,
                    dim_out,
                    kernel_size=4,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=not self.use_bn,
                ))

    def forward(self, image):
        return self.block(image)

class RAE2(nn.Module):
    def __init__(self, in_ch,
                 nb_feature,
                 z_dim,
                 use_bn=True,
                 vae=False,
                 cond=False):
        super().__init__()
        self.use_bn = use_bn
        self.in_ch = in_ch
        self.z_dim = z_dim
        self.out_dim = nb_feature*2*2*8*8 ## nb_feature*2*2*7*7
        self.vae = vae
        self.cond = cond
        if self.cond:
            init_ch = 2*self.in_ch
        else:
            init_ch = self.in_ch
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(init_ch, nb_feature,
                      kernel_size=3,
                      padding=1,
                      bias=not self.use_bn),  # size : (28,28) (32,32)


            RaeBlock(nb_feature, nb_feature*2, ds=True,
                     padding=1,
                     use_bn=self.use_bn),  # size : (14,14) (16, 16)

            nn.ReLU(),
            nn.Conv2d(nb_feature*2, nb_feature*2,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            RaeBlock(nb_feature*2, nb_feature*4, ds=True,
                     padding=1,
                     use_bn=self.use_bn),# size : (7,7) (8,8)

            nn.ReLU(),
            nn.Conv2d(nb_feature * 4, nb_feature * 4,
                      kernel_size=3,
                      padding=1,
                      bias=True),
        )
        if self.vae:
            self.to_mu = nn.Linear(self.out_dim, self.z_dim)
            self.to_logsig = nn.Linear(self.out_dim, self.z_dim)
        else:
            self.to_latent = nn.Linear(self.out_dim, self.z_dim)

        # Decoder
        if self.cond:
            init_z_dec = self.z_dim + 10
        else:
            init_z_dec = self.z_dim
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(init_z_dec, nb_feature*4,
                      kernel_size=8, #7
                      stride=1,
                      padding=0,
                      bias=not self.use_bn),  # size : (7, 7)
            nn.ReLU(),
            nn.Conv2d(nb_feature*4, nb_feature*4,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            RaeBlock(nb_feature*4, nb_feature*2, ds=False,
                     padding=1,
                     use_bn=self.use_bn),  # size : (14, 14)

            nn.ReLU(),
            nn.Conv2d(nb_feature * 2, nb_feature * 2,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            RaeBlock(nb_feature*2, nb_feature, ds=False,
                     padding=1,
                     use_bn=self.use_bn),  # size : (28, 28)

            nn.ReLU(),
            nn.Conv2d(nb_feature, nb_feature,
                      kernel_size=3,
                      padding=1,
                      bias=True),

        )

        self.to_out = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(
                    nb_feature,
                    self.in_ch,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                ),
                nn.Tanh()
        )

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, image, y=None):
        if y is not None and self.cond:
            y = torch.ones_like(image)*y[:,None,None,None]
            image = torch.cat((image, y), dim=1)

        z_int = self.encoder(image)
        z_int = z_int.view(z_int.size(0), -1)
        if self.vae:
            z_mu = self.to_mu(z_int)
            z_logvar = self.to_logsig(z_int)
            return z_mu, z_logvar
        else:
            z = self.to_latent(z_int)
            return z

    def decode(self, latent, y=None):
        if y is not None and self.cond:
            y = F.one_hot(y, 10)
            latent = torch.cat([latent, y], dim=1)
        out1 = self.decoder(latent[:, :, None, None])
        return self.to_out(out1)

    def forward(self, input, y=None, sample=True):
        if self.vae:
            mu, log_var = self.encode(input, y)
            if sample:
                return self.sample(mu, log_var)
            else:
                return mu
        else:
            return self.encode(input, y)


class RAE_conv(nn.Module):
    def __init__(self, in_ch,
                 nb_feature,
                 z_dim,
                 use_bn=True,
                 vae=False,
                 cond=False):
        super().__init__()
        self.use_bn = use_bn
        self.in_ch = in_ch
        self.z_dim = z_dim
        self.out_dim = nb_feature*2*2*8*8 ## nb_feature*2*2*7*7
        self.vae = vae
        self.cond = cond
        if self.cond:
            init_ch = 2*self.in_ch
        else:
            init_ch = self.in_ch
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(init_ch, nb_feature,
                      kernel_size=3,
                      padding=1,
                      bias=not self.use_bn),  # size : (28,28) (32,32)


            RaeBlock(nb_feature, nb_feature*2, ds=True,
                     padding=1,
                     use_bn=self.use_bn),  # size : (14,14) (16, 16)

            nn.ReLU(),
            nn.Conv2d(nb_feature*2, nb_feature*2,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            RaeBlock(nb_feature*2, nb_feature*4, ds=True,
                     padding=1,
                     use_bn=self.use_bn),# size : (7,7) (8,8)

            nn.ReLU(),
            nn.Conv2d(nb_feature * 4, nb_feature * 4,
                      kernel_size=3,
                      padding=1,
                      bias=True),
        )
        if self.vae:
            self.to_mu = nn.Conv2d(nb_feature * 4, 4,
                      kernel_size=3,
                      padding=1,
                      bias=True)
            self.to_logsig = nn.Conv2d(nb_feature * 4, 4,
                      kernel_size=3,
                      padding=1,
                      bias=True)
        else:
            self.to_latent = nn.Conv2d(nb_feature * 4, 4,
                      kernel_size=3,
                      padding=1,
                      bias=True)

        # Decoder
        if self.cond:
            init_z_dec = self.z_dim + 10
        else:
            init_z_dec = self.z_dim
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, nb_feature * 4,
                               kernel_size=1,  # 7
                               stride=1,
                               padding=0,
                               bias=not self.use_bn),  # size : (7, 7)
            nn.ReLU(),
            #nn.ConvTranspose2d(nb_feature*4, nb_feature*4,
            #          kernel_size=8, #7
            #          stride=1,
            #          padding=0,
            #          bias=not self.use_bn),  # size : (7, 7)
            #nn.ReLU(),
            nn.Conv2d(nb_feature*4, nb_feature*4,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            RaeBlock(nb_feature*4, nb_feature*2, ds=False,
                     padding=1,
                     use_bn=self.use_bn),  # size : (14, 14)

            nn.ReLU(),
            nn.Conv2d(nb_feature * 2, nb_feature * 2,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            RaeBlock(nb_feature*2, nb_feature, ds=False,
                     padding=1,
                     use_bn=self.use_bn),  # size : (28, 28)

            nn.ReLU(),
            nn.Conv2d(nb_feature, nb_feature,
                      kernel_size=3,
                      padding=1,
                      bias=True),

        )

        self.to_out = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(
                    nb_feature,
                    self.in_ch,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                ),
                nn.Tanh()
        )

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, image, y=None):
        if y is not None and self.cond:
            y = torch.ones_like(image)*y[:,None,None,None]
            image = torch.cat((image, y), dim=1)

        z_int = self.encoder(image)
        #z_int = z_int.view(z_int.size(0), -1)
        if self.vae:
            z_mu = self.to_mu(z_int)
            z_logvar = self.to_logsig(z_int)
            return z_mu, z_logvar
        else:
            z = self.to_latent(z_int)
            return z

    def decode(self, latent, y=None):
        if y is not None and self.cond:
            y = F.one_hot(y, 10)
            latent = torch.cat([latent, y], dim=1)
        out1 = self.decoder(latent)
        return self.to_out(out1)

    def forward(self, input, y=None, sample=True):
        if self.vae:
            mu, log_var = self.encode(input, y)
            if sample:
                return self.sample(mu, log_var)
            else:
                return mu
        else:
            return self.encode(input, y)

class RAE(nn.Module):
    def __init__(self, in_ch, ch_list, z_dim, use_bn=True, vae=False):
        super().__init__()
        self.use_bn = use_bn
        self.in_ch = in_ch
        self.ch_list = ch_list
        self.z_dim = z_dim
        self.out_dim = ch_list[2]*3*3
        self.vae = vae

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_ch, self.ch_list[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=not self.use_bn),  # size : (14,14)
            RaeBlock(self.ch_list[0], self.ch_list[1], ds=True,
                     padding=1,
                     use_bn=self.use_bn),  # size : (7,7)
            RaeBlock(self.ch_list[1], self.ch_list[2], ds=True,
                     padding=1,
                     use_bn=self.use_bn),  # size : (3,3)
        )
        if self.vae:
            self.to_mu = nn.Linear(self.out_dim, self.z_dim)
            self.to_logsig = nn.Linear(self.out_dim, self.z_dim)
        else:
            self.to_latent = nn.Linear(self.out_dim, self.z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, self.ch_list[2],
                      kernel_size=7,
                      stride=1,
                      padding=0,
                      bias=not self.use_bn),  # size : (7, 7)
            RaeBlock(self.ch_list[2], self.ch_list[1], ds=False,
                     padding=1,
                     use_bn=self.use_bn),  # size : (14, 14)
            RaeBlock(self.ch_list[1], self.ch_list[0], ds=False,
                     padding=1,
                     use_bn=self.use_bn),  # size : (28, 28)
        )

        self.to_out = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(
                    self.ch_list[0],
                    self.in_ch,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                ),
                nn.Sigmoid(),
        )

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, image):
        z_int = self.encoder(image)
        z_int = z_int.view(z_int.size(0), -1)
        if self.vae:
            z_mu = F.relu(self.to_mu(z_int))
            z_logvar = F.relu(self.to_logsig(z_int))
            return z_mu, z_logvar
        else:
            z = self.to_latent(z_int)
            return z

    def decode(self, latent):
        out1 = self.decoder(latent[:, :, None, None])
        return self.to_out(out1)

    def forward(self, input, sample=True):
        if self.vae :
            mu, log_var = self.encode(input)
            if sample :
                return self.sample(mu, log_var)
            else:
                return mu
        else:
            return self.encode(input)

def normalize(z, stats, unormalize=False):
    if unormalize:
        return z * stats['std'] + stats['mean']
    else:
        return (z - stats['mean']) / stats['std']

def normalize_1(z, stats, unormalize=False):
    if unormalize:
        return 0.5*(z + 1)*(stats['max']-stats['min']) + stats['min']
    else:
        return 2*((z - stats['min'])/(stats['max']-stats['min']))-1

def normalize_01(z, stats, unormalize=False):
    if unormalize:
        return z * (stats['max']-stats['min']) + stats['min']
    else:
        return (z - stats['min'])/(stats['max']-stats['min'])