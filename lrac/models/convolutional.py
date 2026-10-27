import torch
from torch import nn
from torch.nn import functional as F
from lrac.models.base import ReconstructiveModel, budget
from lrac.sparse import fista


class ConvolutionalDictionary(ReconstructiveModel):
    is_dictionary = True

    def __init__(self, config, sample_rate):
        super().__init__()
        self.config, self.sample_rate = config, sample_rate
        self.kernels = nn.Parameter(torch.randn(config.conv_kernels, 1, config.conv_length), requires_grad=config.trainable_kernels)
        self.padding = (config.conv_length - config.conv_hop) // 2
        self.post_step()

    def matrix(self):
        return self.kernels[:, 0].T

    def encode(self, x, lengths=None):
        hop = self.config.conv_hop
        extra = (-x.shape[-1]) % hop
        target = F.pad(x, (0, extra))
        weights = self.kernels.detach()
        def synthesis(a):
            return F.conv_transpose1d(a.transpose(1, 2), weights, stride=hop, padding=self.padding)
        def analysis(y):
            return F.conv1d(y, weights, stride=hop, padding=self.padding).transpose(1, 2)
        shape = analysis(target).shape
        codes = fista(target.detach(), analysis, synthesis, shape, self.config.threshold, self.config.inference_steps, self.config.inference_tolerance, self.config.group_threshold, self.config.groups)
        codes = budget(codes, self.config.code_budget)
        return self.make_representation(codes, x, lengths, {"hop": hop, "right": extra})

    def decode(self, representation):
        output = F.conv_transpose1d(representation.concatenate().transpose(1, 2), self.kernels, stride=self.config.conv_hop, padding=self.padding)
        return output[..., :representation.original_samples]

    @torch.no_grad()
    def post_step(self):
        self.kernels.div_(self.kernels.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-12))
