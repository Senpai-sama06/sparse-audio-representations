"""Explicit entry point for the optional GS parameter-predictor stage."""

from approaches.pitchflower.training.campaign import campaign


def train_predictor(config):
    """Train the predictor against the saved GS codec and its latent targets."""
    return campaign(config, phase="predictor")
