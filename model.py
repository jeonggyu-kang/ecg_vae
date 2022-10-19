import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from operator import mul
import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# class Interpolate(nn.Module):
#     def __init__(self, factor):
#         super(Interpolate, self).__init__()
#         self.factor = factor
#         return

#     def forward(self, input):
#         return F.interpolate(input,
#                              scale_factor=self.factor,
#                              mode='linear',
#                              align_corners=True)

class Unflatten(nn.Module):
    def __init__(self, channel, length):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.length  = length

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.length)

class ResBlock(nn.Module):
    def __init__(self, n_input, n_output, kernel_size):
        super(ResBlock, self).__init__()

        self.conv1     = nn.Conv1d(n_input, n_output, kernel_size=kernel_size,
                                   padding='same', bias=False)
        self.conv2     = nn.Conv1d(n_output, n_output, kernel_size=kernel_size,
                                   padding='same', bias=False)
        self.relu     = nn.ReLU(inplace=True)
        self.bn1      = nn.BatchNorm1d(n_output)
        self.bn2      = nn.BatchNorm1d(n_output)
        self.n_input  = n_input
        self.n_output = n_output

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.n_input == self.n_output:
            out = out + x
        elif self.n_input < self.n_output:
            n_factor = int(self.n_output/self.n_input)
            out      = out + x.repeat(1, n_factor, 1)
        return out

class ResUpsampleBlock(nn.Module):
    def __init__(self, n_input, n_output, upsample_rate, kernel_size):
        super(ResUpsampleBlock, self).__init__()
        self.upsample_rate = upsample_rate
        self.conv1   = nn.Conv1d(n_input, n_output, kernel_size=kernel_size,
                                 padding='same', bias=False)

        self.conv2   = nn.Conv1d(n_output, n_output, kernel_size=kernel_size,
                                 padding='same', bias=False)

        self.relu     = nn.ReLU(inplace=True)
        self.bn1      = nn.BatchNorm1d(n_output)
        self.bn2      = nn.BatchNorm1d(n_output)
        self.n_input  = n_input
        self.n_output = n_output

    def forward(self, x):
        x   = F.interpolate(x,
                            scale_factor=self.upsample_rate,
                            mode='linear',
                            align_corners=True)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        x   = x[:, 0:out.size()[1], :]
        out = out + x
        return out

class ConvVAE(nn.Module):
    def __init__(self, latent_size):
        super(ConvVAE, self).__init__()

        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, padding='same', bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            # 5000 -> 2500
            ResBlock(8,  8, kernel_size=9),

            nn.Conv1d(8, 8, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            # 2500 -> 1250 
            ResBlock(8,  16, kernel_size=9),

            nn.Conv1d(16, 16, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # 1250 -> 625
            ResBlock(16, 32, kernel_size=9),

            nn.Conv1d(32, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # 625 -> 125
            ResBlock(32, 64, kernel_size=9),

            nn.Conv1d(64, 64, kernel_size=9, stride=5, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # 125 -> 25
            ResBlock(64,  128, kernel_size=3),

            nn.Conv1d(128, 128, kernel_size=3, stride=5, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            Flatten(),
        )

        # hidden => mu
        self.fc1 = nn.Linear(25*128, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(25*128, self.latent_size)

        self.decoder = nn.Sequential(
            Unflatten(self.latent_size, 1),

            # 1 -> 25
            nn.ConvTranspose1d(self.latent_size, 128, kernel_size=25, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            ResBlock(128, 128, kernel_size=3),

            # 25 -> 125
            ResUpsampleBlock(128, 64, 5, 3),

            # 125 -> 625
            ResUpsampleBlock(64, 32, 5, 9),

            # 625 -> 1250 
            ResUpsampleBlock(32, 16, 2, 9),

            # 1250 -> 2500
            ResUpsampleBlock(16, 8, 2, 9),

            # 2500 -> 5000
            ResUpsampleBlock(8, 8, 2, 9),

            # 5000 -> 5000
            nn.Conv1d(8, 1, kernel_size=9, padding='same'),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        logvar = torch.clamp(logvar, min=-20, max=20)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def reparameterize_eval(self, mu, logvar, n_samples=None):
        std = torch.exp(0.5 * logvar)
        eps = None
        if n_samples is None:
            eps = torch.randn_like(std)
        else:
            eps = torch.randn([n_samples, self.latent_size],
                              device=std.get_device())
        return eps.mul(std).add_(mu), eps

    def forward(self, *args, **kwargs):
        if len(kwargs) == 0:
            x = args[0]
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar
        elif kwargs["op"] == "encode":
            x = args[0]
            return self.encode(x)
        elif kwargs["op"] == "decode":
            z = args[0]
            return self.decode(z)
        elif kwargs["op"] == "reparam":
            mu     = args[0]
            logvar = args[1]
            return self.reparameterize(mu, logvar)
        elif kwargs["op"] == "reparam_eval":
            mu        = args[0]
            logvar    = args[1]
            n_samples = kwargs.get("n_samples", None)
            return self.reparameterize_eval(mu, logvar, n_samples)
        return
