"""Freeze the dependency closure of this project, excluding unrelated global apps."""
from importlib import metadata
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOTS = ["torch", "numpy", "scipy", "soundfile", "librosa", "scikit-learn", "PyYAML", "matplotlib", "tensorboard", "pytest", "setuptools", "wheel"]


def main():
    pending, versions = list(ROOTS), {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in versions:
            continue
        distribution = metadata.distribution(name)
        versions[name] = distribution.version
        for text in distribution.requires or []:
            req = Requirement(text)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                pending.append(req.name)
    output = Path(__file__).resolve().parents[1] / "requirements-lock.txt"
    lines = ["# Tested Windows/Python 3.11 dependency closure; regenerate with tools/lock_environment.py", "--extra-index-url https://download.pytorch.org/whl/cu128"]
    lines += [f"{name}=={version}" for name, version in sorted(versions.items())]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Locked {len(versions)} distributions to {output}")


if __name__ == "__main__":
    main()
