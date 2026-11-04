"""Shared function decoders: transmitted codes are separate from shared weights.

Coordinates and code length determine a 640-sample function. Both neural
candidates use the same trainable Fourier features and FiLM architecture; only
their hidden activation differs. Prefix masking permits explicit code budgets.
"""
import torch
from torch import nn


class PeriodicActivation(nn.Module):
    def __init__(self,width,kind):
        super().__init__();self.kind=kind
        if kind=='snakebeta':
            self.log_alpha=nn.Parameter(torch.zeros(width));self.log_beta=nn.Parameter(torch.zeros(width))

    def forward(self,x):
        if self.kind=='sine':return x.sin()
        alpha=self.log_alpha.clamp(-6,6).exp();beta=self.log_beta.clamp(-6,6).exp()
        return x+(alpha*x).sin().square()/beta


class SharedDecoder(nn.Module):
    """A conditional mathematical function of time; no source samples at decode."""
    def __init__(self,activation='sine',max_dim=64,width=64,features=32):
        super().__init__();self.spec=dict(activation=activation,max_dim=max_dim,width=width,features=features)
        self.max_dim=max_dim
        self.frequencies=nn.Parameter(torch.linspace(1,torch.pi*639/2,features))
        self.register_buffer('time',torch.linspace(-1,1,640))
        self.input=nn.Linear(2*features+1,width)
        self.layers=nn.ModuleList([nn.Linear(width,width) for _ in range(2)])
        self.condition=nn.ModuleList([nn.Linear(max_dim,2*width) for _ in range(3)])
        self.activations=nn.ModuleList([PeriodicActivation(width,activation) for _ in range(3)])
        self.output=nn.Linear(width,1)
        for layer in self.condition:
            nn.init.normal_(layer.weight,std=.01);nn.init.zeros_(layer.bias)

    def forward(self,code):
        phase=self.time[:,None]*self.frequencies[None,:]
        basis=torch.cat([self.time[:,None],phase.sin(),phase.cos()],dim=-1)
        h=self.input(basis)[None,:,:].expand(len(code),-1,-1)
        for i,(condition,activation) in enumerate(zip(self.condition,self.activations)):
            if i:h=self.layers[i-1](h)
            gamma,beta=condition(code).chunk(2,dim=-1)
            h=activation(h*(1+.1*gamma[:,None,:])+beta[:,None,:])
        return self.output(h).squeeze(-1)


class SharedRepresentation(nn.Module):
    def __init__(self,activation='sine',max_dim=64,width=64,features=32):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(640,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,max_dim))
        self.decoder=SharedDecoder(activation,max_dim,width,features)

    def encode(self,x,dimensions):
        code=self.encoder(x)
        mask=torch.arange(code.shape[-1],device=code.device)<dimensions
        return code*mask

    def forward(self,x,dimensions):
        code=self.encode(x,dimensions);return self.decoder(code),code


def normalize(x):
    """Mean and gain are two explicit per-component transmitted scalars."""
    mean=x.mean(-1,keepdim=True);gain=(x-mean).square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
    return (x-mean)/gain,mean,gain
