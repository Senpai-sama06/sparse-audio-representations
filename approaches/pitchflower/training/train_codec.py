"""Explicit entry point for training an RVQ or GS PitchFlower codec stage."""

from approaches.pitchflower.training.campaign import campaign


def train_codec(config, bottleneck):
    """Run one codec training stage using the shared PitchFlower campaign.

    Args:
        config: Resolved study settings for the PitchFlower campaign.
        bottleneck: ``"rvq"`` or ``"gs"``; selects the codec stage and output.
    """
    if bottleneck not in {"rvq", "gs"}:
        raise ValueError("bottleneck must be 'rvq' or 'gs'")
    return campaign(config, phase=bottleneck)
