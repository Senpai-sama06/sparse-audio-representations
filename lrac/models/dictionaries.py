import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal
from lrac.models.base import ReconstructiveModel, budget
from lrac.models.transforms import dct_basis, frame, unframe
from lrac.sparse import fista, sparse_group_prox


class FrameDictionary(ReconstructiveModel):
    is_dictionary = True

    def __init__(self, config, sample_rate):
        super().__init__()
        self.config, self.sample_rate = config, sample_rate
        n = config.frame_length
        self.register_buffer("window", torch.hann_window(n).sqrt())
        self.register_buffer("frame_mean", torch.zeros(n))
        self.needs_fit = config.kind == "pca"
        if config.kind in {"orthogonal", "grouped", "overcomplete"}:
            m = config.atoms if config.kind == "overcomplete" else n
            self.dictionary = nn.Linear(m, n, bias=False)
            if m == n:
                with torch.no_grad():
                    self.dictionary.weight.copy_(dct_basis(n))
                orthogonal(self.dictionary, "weight", orthogonal_map="householder")
            else:
                self.post_step()
        else:
            matrix = dct_basis(n) if config.kind in {"dct", "pca"} else torch.linalg.qr(torch.randn(n, n))[0]
            self.register_buffer("basis", matrix)

    def matrix(self):
        return self.dictionary.weight if hasattr(self, "dictionary") else self.basis

    def encode(self, x, lengths=None):
        frames, timing = frame(x, self.window, self.config.hop)
        target = frames - self.frame_mean
        matrix = self.matrix()
        if self.config.kind == "overcomplete":
            detached = matrix.detach()
            codes = fista(target.detach(), lambda y: y @ detached, lambda a: a @ detached.T, (*target.shape[:-1], detached.shape[1]), self.config.threshold, self.config.inference_steps, self.config.inference_tolerance, self.config.group_threshold, self.config.groups)
        else:
            codes = target @ matrix
            codes = sparse_group_prox(codes, self.config.threshold, self.config.group_threshold, self.config.groups)
            if self.training and hasattr(self, "dictionary"):
                codes = codes.detach()  # Alternating dictionary update, not an unrolled encoder.
        codes = budget(codes, self.config.code_budget)
        return self.make_representation(codes, x, lengths, timing)

    def decode(self, representation):
        frames = representation.concatenate() @ self.matrix().T + self.frame_mean
        return unframe(frames, self.window, representation.timing, representation.original_samples)

    @torch.no_grad()
    def post_step(self):
        if hasattr(self, "dictionary") and self.config.kind == "overcomplete":
            self.dictionary.weight.div_(self.dictionary.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))

    @torch.no_grad()
    def fit(self, loader, device, max_frames):
        n = self.config.frame_length
        total, count = torch.zeros(n, dtype=torch.float64, device=device), 0
        outer = torch.zeros(n, n, dtype=torch.float64, device=device)
        for x, lengths, _ in loader:
            x = x.to(device)
            # Fit only frames fully within valid audio; padded samples cannot bias PCA.
            for item, length in zip(x, lengths):
                valid = item[0, :int(length)]
                if valid.numel() < n:
                    continue
                frames = valid.unfold(-1, n, self.config.hop) * self.window
                frames = frames[:max_frames - count].double()
                total += frames.sum(0)
                outer += frames.T @ frames
                count += len(frames)
                if count >= max_frames:
                    break
            if count >= max_frames:
                break
        if count < n:
            raise ValueError(f"PCA requires at least {n} frames; found {count}")
        mean = total / count
        covariance = outer / count - mean[:, None] * mean[None, :]
        _, vectors = torch.linalg.eigh(covariance)
        self.basis.copy_(vectors.flip(-1).to(self.basis.dtype))
        self.frame_mean.copy_(mean.to(self.frame_mean.dtype))
        self.needs_fit = False
