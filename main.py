"""Project entry point for selecting and assembling research runs.

Define Python-configured ablation cases in ``EXPERIMENTS`` below. Each case
should construct its own model/trainer arguments and call an approach-level
training or study function. Leave the list empty to use the legacy CLI.
"""
from dataclasses import dataclass
from typing import Callable

from approaches.runner.cli import main as cli_main


@dataclass(frozen=True)
class ExperimentCase:
    """A named Python-defined run; the callable owns its model and settings."""

    name: str
    action: Callable[[], object]

    def execute(self):
        """Run this case and label its progress in the terminal."""
        print(f"CASE={self.name}", flush=True)
        return self.action()


# Define settings and cases here for direct Python-driven ablations. For
# example, construct RVQ/GS settings as Python objects and use
# ``ExperimentCase(name, lambda: train_codec(settings, bottleneck))``. The
# current command-line workflows remain the default until cases are listed.
EXPERIMENTS: list[ExperimentCase] = []

# Shared learned-function pilot for experiment 1 (fixed VMD; no RVQ):
# from approaches.parametric.learned import run as train_shared_functions
# EXPERIMENTS = [ExperimentCase("learned_mode_functions", lambda: train_shared_functions(
#     corpus_root="data/LibriSpeech", output="runs/parametric_learned_representation",
#     steps=12000, budgets=(4,8,16,32,64),
# ))]

# Isolated 40 ms capacity study: exactly one 640-sample block per testcase,
# no overlap/stitching/quantization. Offsets are fixed before fitting.
# from approaches.parametric.isolated import run as run_isolated
# EXPERIMENTS = [ExperimentCase("isolated_40ms", lambda: run_isolated(
#     source="data/testcase.wav", output="runs/parametric_isolated_40ms",
#     offsets=(.96, 1.96, 2.96), restarts=3, max_steps=1600,
# ))]

# Full parametric coding grid (or call its module CLI):
# from approaches.parametric.ablation import run_ablation
# EXPERIMENTS = [ExperimentCase("parametric_coding", lambda: run_ablation(
#     audio="data/testcase.wav", output="runs/parametric_ablation",
#     windows=(40, 20, 80, 160), budgets=(500, 1000, 2000, 4000, 8000),
#     trials=32, steps=80,
# ))]

# Bare EMD/VMD testcase sweep, with settings defined directly in Python:
# from approaches.parametric import run as run_parametric
# EXPERIMENTS = [ExperimentCase("parametric_windows", lambda: run_parametric(
#     "data/my_speech.wav", window_ms=(20, 40, 80, 160), vmd_modes=8,
# ))]


def main():
    """Execute selected Python cases, or fall back to the existing CLI."""
    names = [case.name for case in EXPERIMENTS]
    if len(names) != len(set(names)):
        raise ValueError("Experiment case names must be unique")
    if EXPERIMENTS:
        for case in EXPERIMENTS:
            case.execute()
        return
    cli_main()


if __name__ == "__main__":
    main()
