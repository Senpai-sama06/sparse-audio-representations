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
