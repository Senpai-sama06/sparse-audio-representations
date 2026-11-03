"""Compact mathematical mode models. Decoding depends only on model parameters.

Time is always normalized to [-1, 1]. Fits minimize waveform MSE, never PESQ.
Neural fits are per-mode optimization; every fitted weight is in the payload.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
import torch

FAMILIES = ('polynomial', 'spline', 'sinusoid', 'amfm', 'sine_network', 'snakebeta_network')


@dataclass
class ModeModel:
    family: str
    parameters: np.ndarray

    def decode(self, samples):
        return synthesize(self.family, self.parameters, samples)


def _network(p, t, snake):
    """Two hidden layers; trainable SnakeBeta alpha/beta are log parameters."""
    # Infer width from exact parameter count to avoid transmitting architecture.
    h = next(k for k in range(1, 129) if k*k + (8 if snake else 4)*k + 1 == len(p))
    i = 0
    def take(n):
        nonlocal i
        z = p[i:i+n]; i += n
        return z
    w1, b1 = take(h), take(h)
    w2, b2, w3, b3 = take(h*h).reshape(h, h), take(h), take(h), take(1)
    z = t[:, None] * w1 + b1
    if snake:
        a1, beta1, a2, beta2 = [take(h).clamp(-6, 6).exp() for _ in range(4)]
        z = z + (a1*z).sin().square()/beta1
        z = z @ w2 + b2
        z = z + (a2*z).sin().square()/beta2
    else:
        z = z.sin(); z = (z @ w2 + b2).sin()
    return z @ w3 + b3


def synthesize(family, p, samples):
    """Pure decoder: no target waveform, fitting state, or normalization cache."""
    p = np.asarray(p, dtype=np.float64)
    t = np.linspace(-1, 1, samples)
    if len(p) == 0:
        return np.zeros(samples)
    if family == 'polynomial':
        return np.polynomial.chebyshev.chebval(t, p)
    if family == 'spline':
        if len(p) == 1:
            return np.full(samples, p[0])
        return CubicSpline(np.linspace(-1, 1, len(p)), p, bc_type='natural')(t)
    if family == 'sinusoid':
        a, f, phase = p.reshape(-1, 3).T
        return (a[:, None]*np.cos(f[:, None]*t + phase[:, None])).sum(0)
    if family == 'amfm':
        n = len(p)//2
        amplitude = np.polynomial.polynomial.polyval(t, p[:n])
        phase = np.polynomial.polynomial.polyval(t, p[n:])
        return np.maximum(amplitude, 0)*np.cos(phase)
    if family.endswith('_network'):
        with torch.no_grad():
            return _network(torch.tensor(p), torch.tensor(t), family.startswith('snake')).numpy()
    raise ValueError(family)


def fit_mode(x, family, budget, steps=80, seed=42):
    """Fit within the scalar budget; unrepresentable architectures use zero values.

    Sinusoidal atoms use residual FFT peak search and a joint least-squares fit.
    Nonlinear families retain the best finite iterate, including initialization.
    """
    x = np.asarray(x, dtype=np.float64)
    budget = min(int(budget), len(x), 256)
    t = np.linspace(-1, 1, len(x))
    if budget <= 0 or np.max(np.abs(x)) < 1e-12:
        return ModeModel(family, np.array([]))
    if family == 'polynomial':
        p = np.polynomial.chebyshev.chebfit(t, x, min(budget, 128)-1)
    elif family == 'spline':
        n = min(budget, 128)
        if n == 1:
            p = np.array([x.mean()])
        else:
            basis = CubicSpline(np.linspace(-1, 1, n), np.eye(n), bc_type='natural')(t)
            p = np.linalg.lstsq(basis, x, rcond=None)[0]
    elif family == 'sinusoid':
        residual, frequencies, basis = x.copy(), [], np.empty((len(x), 0))
        for _ in range(budget//3):
            spec = np.abs(np.fft.rfft(residual, n=4*len(x)))
            f = np.argmax(spec)*2*np.pi/(4*len(x))*(len(x)-1)/2
            frequencies.append(f)
            basis = np.column_stack([basis, np.cos(f*t), np.sin(f*t)])
            weights = np.linalg.lstsq(basis, x, rcond=None)[0]
            residual = x - basis @ weights
        p = np.array([[np.hypot(a,b), f, np.arctan2(-b,a)]
                      for f,(a,b) in zip(frequencies, weights.reshape(-1,2))]).ravel() if frequencies else np.array([])
    else:
        scale = max(np.std(x), 1e-8)
        target = torch.tensor(x/scale, dtype=torch.float64)
        tt = torch.tensor(t, dtype=torch.float64)
        torch.manual_seed(seed)
        if family == 'amfm':
            n = min(budget//2, 8)
            if n < 2:
                return ModeModel(family, np.array([]))
            f = np.argmax(np.abs(np.fft.rfft(x)))*np.pi*(len(x)-1)/len(x)
            b = np.column_stack([np.cos(f*t), np.sin(f*t)])
            a, c = np.linalg.lstsq(b, x/scale, rcond=None)[0]
            init = np.zeros(2*n); init[0] = np.hypot(a,c)
            init[n], init[n+1] = np.arctan2(-c,a), f
            def forward(p):
                powers = tt[:,None]**torch.arange(n)[None,:]
                return (powers@p[:n]).clamp_min(0)*(powers@p[n:]).cos()
        else:
            snake = family.startswith('snake')
            widths = [h for h in range(1, 16) if h*h+(8 if snake else 4)*h+1 <= budget]
            if not widths:
                return ModeModel(family, np.array([]))
            h = max(widths); init = np.random.default_rng(seed).normal(0, .3, h*h+(8 if snake else 4)*h+1)
            init[:h] = np.linspace(1, min(len(x)*np.pi/2, 100), h)
            if snake:
                init[-4*h:] = 0
            def forward(p):
                return _network(p, tt, snake)
        ptorch = torch.tensor(init, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([ptorch], lr=.02)
        best, best_loss = ptorch.detach().numpy().copy(), float('inf')
        for _ in range(steps+1):
            optimizer.zero_grad(); loss = (forward(ptorch)-target).square().mean()
            if not torch.isfinite(loss):
                break
            if loss.item() < best_loss:
                best_loss, best = loss.item(), ptorch.detach().numpy().copy()
            loss.backward(); torch.nn.utils.clip_grad_norm_([ptorch], 10); optimizer.step()
        p = best
        if family == 'amfm':
            p[:len(p)//2] *= scale
        else:
            h = max(widths); offset = h*h+3*h
            p[offset:offset+h+1] *= scale
    return ModeModel(family, np.asarray(p, dtype=np.float64))
