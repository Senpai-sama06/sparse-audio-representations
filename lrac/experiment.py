"""One readable validation workflow using the individual commands' functions."""
from lrac.data.manifests import prepare
from lrac.evaluate import evaluate
from lrac.analyze import analyze
from lrac.report import report
from lrac.train import train


def run_experiment(config, download=False, limit=None, skip_prepare=False):
    """Prepare, audit/train, evaluate validation, analyze, and write a report.

    Final test sets are left for a separate evaluate command. Downloads require
    an explicit request.
    """
    if config.data.source == "vctk" and config.data.vctk_mode == "external":
        raise ValueError("VCTK external mode is evaluation-only; use evaluate --data-config")
    if limit is not None and limit <= 0:
        raise ValueError("Validation limit must be positive")

    if skip_prepare and download:
        raise ValueError("--skip-prepare cannot be combined with --download")
    if skip_prepare:
        print("[1/5] Use prepared manifests (training still audits them)", flush=True)
    else:
        print("[1/5] Prepare data manifests", flush=True)
        prepare(config, download=download)

    print("[2/5] Audit data and train", flush=True)
    run = train(config)

    validation_limit = limit if limit is not None else config.train.val_examples
    print("[3/5] Evaluate validation examples", flush=True)
    evaluate(run, split="validation", limit=validation_limit, device=config.train.device)

    print("[4/5] Analyze validation features", flush=True)
    analyze(run, split="validation", limit=min(validation_limit, config.evaluation.analysis_examples), device=config.train.device)

    print("[5/5] Write report", flush=True)
    output = report([run], run / "report")
    print(f"Open {output / 'index.html'}", flush=True)
    return run
