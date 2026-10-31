"""Fetch immutable official sources; retain their licenses in ignored runs/vendor.

Run once before PitchFlower training. Optional Python dependencies are declared
in pyproject.toml; existing CUDA torch/torchaudio installations are not replaced.
"""
from pathlib import Path
import subprocess
import sys
from importlib.util import find_spec
import argparse

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    "PitchFlower": ("https://github.com/diegotg2000/PitchFlower.git", "4cef72ac9630cab632f13bd1d26b22cdae81cf17"),
    "gs-codec": ("https://github.com/ronaluf/gs-codec.git", "c5261bbc3f90838dc21b3b134ceadd15d702f007"),
}


def setup(vocoder=False):
    """Pin both reference checkouts; the comparison imports only PitchFlower."""
    for name, (url, revision) in REFERENCES.items():
        destination = ROOT/"runs/vendor"/name
        if not destination.exists():
            subprocess.run(["git", "clone", url, str(destination)], check=True)
        actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            subprocess.run(["git", "-C", str(destination), "fetch", "origin", revision], check=True)
            # Refuse to overwrite local upstream edits while selecting the pin.
            if subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True).strip():
                raise RuntimeError(f"Local reference edits in {destination}")
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
        if subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
            raise RuntimeError(f"Tracked reference source edits in {destination}; restore the pinned source before running")
        print(name, revision)
    optional = {"pytorch_lightning": "pytorch-lightning==2.5.5", "torchaudio": "torchaudio==2.8.0",
        "einops": "einops==0.8.1", "rotary_embedding_torch": "rotary-embedding-torch==0.8.9",
        "torchdyn": "torchdyn==1.0.6", "vocos": "vocos==0.1.0", "pyworld": "pyworld==0.3.5",
        "tqdm": "tqdm>=4.67", "tensorboard": "tensorboard>=2.18"}
    missing = [package for module, package in optional.items() if find_spec(module) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing,
                        "--extra-index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    if vocoder:
        # Fail on missing/inaccessible inference assets before spending days on
        # preprocessing or training. This loads pretrained weights on CPU only.
        from vocos import Vocos
        model = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
        del model
    print("References and optional dependencies ready; no training was started.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocoder", action="store_true", help="Also fetch/check the fixed Vocos weights; never train")
    setup(parser.parse_args().vocoder)
