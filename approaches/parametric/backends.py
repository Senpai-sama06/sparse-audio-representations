"""Additional independently windowed backends; no mode identities cross frames."""
import numpy as np
from .decomposition import Decomposition, ModeDecomposer, EMDDecomposer, VMDDecomposer


class EnsembleDecomposer(ModeDecomposer):
    """Pinned PyEMD EEMD or CEEMDAN implementation with reproducible ensembles.

    PyEMD CEEMDAN includes later algorithm improvements; the precise package
    version and constructor settings are stored in the experiment manifest.
    """
    def __init__(self, name, trials=32, seed=42):
        from PyEMD import EEMD, CEEMDAN
        self.name, self.seed = name, seed
        cls = EEMD if name == 'eemd' else CEEMDAN
        self.backend = cls(trials=trials, parallel=False)

    def decompose(self, signal):
        x = self.signal(signal)
        if np.std(x) < 1e-12:
            return Decomposition(np.zeros((0, len(x))), x.copy()).validate(len(x))
        self.backend.noise_seed(self.seed)
        self.backend(x)
        modes, residue = self.backend.get_imfs_and_residue()
        return Decomposition(np.asarray(modes).reshape(-1, len(x)), np.asarray(residue),
                             {'implementation': 'PyEMD', 'seed': self.seed}).validate(len(x))


class SuccessiveVMD(ModeDecomposer):
    """PySDKit's SVMD port, bounded by explicit iteration and mode limits.

    Reference: Nazari & Sakhaei (2020), doi:10.1016/j.sigpro.2020.107610.
    Hitting max_modes is recorded rather than interpreted as convergence.
    """
    name = 'svmd'

    def __init__(self, seed=42):
        from pysdkit import SVMD
        self.backend = SVMD(max_modes=12, max_iter=300, random_seed=seed)

    def decompose(self, signal):
        x = self.signal(signal)
        if np.std(x) < 1e-12:
            return Decomposition(np.zeros((0, len(x))), x.copy()).validate(len(x))
        padded = np.pad(x, (0, len(x) % 2), mode='edge')
        modes = np.asarray(self.backend.fit_transform(padded))[:, :len(x)]
        return Decomposition(modes, x - modes.sum(0),
                             {'implementation': 'pysdkit.SVMD', 'mode_limit_reached': len(modes) == 12}).validate(len(x))


def make_backend(name, trials=32, seed=42):
    """Construct explicit research components without configuration machinery."""
    if name == 'emd':
        return EMDDecomposer()
    if name == 'vmd':
        return VMDDecomposer()
    if name in ('eemd', 'ceemdan'):
        return EnsembleDecomposer(name, trials, seed)
    if name == 'svmd':
        return SuccessiveVMD(seed)
    raise ValueError(name)
