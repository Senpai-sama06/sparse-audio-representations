from lrac.data.manifests import write_manifest
from lrac.data.datasets import AudioDataset


def test_bounded_validation_covers_speakers(tmp_path):
    rows = [{"id": f"{speaker}-{i}", "corpus": "example", "speaker": str(speaker)} for speaker in range(4) for i in range(10)]
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, rows)
    dataset = AudioDataset(path, 16000, limit=4)
    assert {r["speaker"] for r in dataset.rows} == {"0", "1", "2", "3"}
