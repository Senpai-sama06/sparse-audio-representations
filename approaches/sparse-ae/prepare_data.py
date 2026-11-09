"""Prepare a common local LibriSpeech manifest set for every baseline."""
import argparse
import json

from sae.data import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-split", default="train-clean-100")
    parser.add_argument("--dev-split", default="dev-clean")
    parser.add_argument("--test-split", default="test-clean")
    parser.add_argument("--train-hours", type=float)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    path = prepare(args.root, args.output, args.train_split, args.dev_split,
                   args.test_split, args.train_hours, args.seed)
    print(f"MANIFESTS={path}")
    print(json.dumps(json.loads((path / "manifest_info.json").read_text(encoding="utf-8"))["counts"], indent=2))


if __name__ == "__main__":
    main()
