import torch
from torch import nn
from torch.nn import functional as F
from lrac.models.base import ReconstructiveModel, budget


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.LeakyReLU(.2), nn.Conv1d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class GroupedAutoencoder(ReconstructiveModel):
    def __init__(self, config, sample_rate):
        super().__init__()
        self.config, self.sample_rate = config, sample_rate
        channels = [1, *config.channels]
        encoder = []
        for source, target in zip(channels[:-1], channels[1:]):
            encoder.extend([nn.Conv1d(source, target, 9, stride=4, padding=4), nn.LeakyReLU(.2), ResidualBlock(target)])
        encoder.append(nn.Conv1d(channels[-1], config.latent_width, 1))
        self.encoder = nn.Sequential(*encoder)
        decoder = [nn.Conv1d(config.latent_width, channels[-1], 1)]
        for source, target in zip(channels[:0:-1], channels[-2::-1]):
            decoder.extend([ResidualBlock(source), nn.LeakyReLU(.2), nn.ConvTranspose1d(source, target, 9, stride=4, padding=4, output_padding=3)])
            if target != 1:
                decoder.append(nn.LeakyReLU(.2))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x, lengths=None):
        extra = (-x.shape[-1]) % 256
        values = self.encoder(F.pad(x, (0, extra))).transpose(1, 2)
        values = budget(values, self.config.code_budget)
        return self.make_representation(values, x, lengths, {"hop": 256, "right": extra})

    def decode(self, representation):
        return self.decoder(representation.concatenate().transpose(1, 2))[..., :representation.original_samples]
