import argparse
import json
from pathlib import Path
import yaml
from lrac.config import load_config, from_dict


def parser():
    p = argparse.ArgumentParser(description="Reconstructive feature discovery for speech and synthetic signals")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["prepare", "audit", "train", "experiment"]:
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "prepare":
            command.add_argument("--download", action="store_true", help="Explicitly acquire official corpus archives")
        elif name == "train":
            command.add_argument("--resume", help="Path to a local last.pt checkpoint")
        elif name == "experiment":
            command.add_argument("--download", action="store_true", help="Explicitly acquire official corpus archives")
            command.add_argument("--limit", type=int, help="Validation examples; defaults to train.val_examples")
            command.add_argument("--skip-prepare", action="store_true", help="Reuse existing manifests; data audit still runs before training")
    for name in ["evaluate", "analyze"]:
        command = sub.add_parser(name)
        command.add_argument("--run", required=True)
        command.add_argument("--split", default="validation" if name == "analyze" else None)
        command.add_argument("--limit", type=int)
        command.add_argument("--device", default="auto")
        command.add_argument("--checkpoint", default="best.pt", choices=["best.pt", "initial.pt", "last.pt"])
        if name == "evaluate":
            command.add_argument("--data-config", help="External dataset configuration; model remains frozen")
            command.add_argument("--budgets", help="Retained coefficients: whole recording for sparse_autoencoder/block_orthogonal, per time step otherwise; zero keeps none in global models, all in legacy models")
            command.add_argument("--keep-fractions", help="Sparse autoencoder: comma-separated fractions retained across each whole recording, including 0 and 1")
            command.add_argument("--metric-workers", type=int, default=0, help="CPU processes for optional PESQ/STOI; zero computes them serially")
            command.add_argument("--resume", action="store_true", help="Reuse complete budgets with matching example coverage and checkpoint metadata")
            command.add_argument("--nmse-targets", help="Block model: comma-separated waveform NMSE tolerances for minimum support")
    command = sub.add_parser("report")
    command.add_argument("--runs", nargs="+", required=True)
    command.add_argument("--output", default="runs/reports/comparison")
    command = sub.add_parser("sweep")
    command.add_argument("--config", required=True, help="Sweep specification with base, axis, and values")
    command.add_argument("--execute", action="store_true", help="Run sequentially; default only writes resolved configs")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if hasattr(args, "limit") and args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.command == "experiment":
        from lrac.experiment import run_experiment
        run_experiment(load_config(args.config), download=args.download, limit=args.limit, skip_prepare=args.skip_prepare)
    elif args.command == "prepare":
        from lrac.data.manifests import prepare
        cfg = load_config(args.config)
        result = prepare(cfg, args.download)
        print(f"MANIFESTS={result}")
    elif args.command == "audit":
        from lrac.data.audit import audit
        cfg = load_config(args.config)
        report = audit(cfg, cfg.path(cfg.data.manifest_dir) / "audit.json")
        print(json.dumps(report, indent=2))
    elif args.command == "train":
        from lrac.train import train
        train(load_config(args.config), args.resume)
    elif args.command == "evaluate":
        from lrac.evaluate import evaluate
        budgets = [int(v) for v in args.budgets.split(",")] if args.budgets else None
        if budgets and min(budgets) < 0:
            raise ValueError("Budgets must be nonnegative")
        targets = [float(v) for v in args.nmse_targets.split(",")] if args.nmse_targets else None
        fractions = [float(v) for v in args.keep_fractions.split(",")] if args.keep_fractions else None
        evaluate(args.run, args.split, args.limit, args.device, args.data_config, budgets, targets, args.checkpoint, fractions, args.metric_workers, args.resume)
    elif args.command == "analyze":
        from lrac.analyze import analyze
        analyze(args.run, args.split, args.limit, args.device, args.checkpoint)
    elif args.command == "report":
        from lrac.report import report
        report(args.runs, args.output)
    elif args.command == "sweep":
        from lrac.train import train
        path = Path(args.config).resolve()
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        allowed = {"base", "axis", "values", "output"}
        if set(spec) - allowed:
            raise ValueError("Sweep permits a single explicit axis; create separate sweeps for different hypotheses")
        base = load_config(path.parent / spec["base"])
        section, field = spec["axis"].split(".")
        destination = base.path(spec.get("output", "runs/resolved_sweeps")) / path.stem
        destination.mkdir(parents=True, exist_ok=True)
        for index, value in enumerate(spec["values"]):
            raw = base.to_dict()
            raw[section][field] = value
            raw["train"]["name"] += f"_{field}_{value}"
            cfg = from_dict(raw)
            generated = destination / f"{index:03d}.yaml"
            generated.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
            print(f"CONFIG={generated}")
            if args.execute:
                train(cfg)
