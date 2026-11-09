from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "LibriSpeech"
    for split, speakers in (("train-clean-100", [1, 2]), ("dev-clean", [3, 4, 5, 6]), ("test-clean", [7, 8])):
        for speaker in speakers:
            directory = root / split / str(speaker) / "10"
            directory.mkdir(parents=True)
            for utterance in range(2):
                t = np.arange(1024 + utterance * 17) / 16000
                audio = .08 * np.sin(2 * np.pi * (190 + speaker * 37 + utterance) * t)
                sf.write(directory / f"{speaker}-10-{utterance:04d}.flac", audio, 16000)
    return root


@pytest.fixture
def manifests(corpus, tmp_path):
    from sae.data import prepare
    path = tmp_path / "manifests"
    prepare(corpus, path)
    return path
