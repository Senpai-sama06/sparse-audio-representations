"""Windowed mode-decomposition baselines for future parametric audio coding."""


def run(*args, **kwargs):
    """Run the EMD/VMD window sweep with ordinary Python arguments."""
    from approaches.parametric.experiment import run as run_experiment
    return run_experiment(*args, **kwargs)
