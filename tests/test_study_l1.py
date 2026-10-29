"""A small public-workflow check plus independent source-selection assertions."""
import csv
import json

import numpy as np
import pytest
import soundfile as sf

from lrac.config import StudyConfig
from lrac.data.l1_examples import speech_examples
from lrac.study import run_study
from lrac.study_l1 import operating_points


def test_selection_excludes_sources_and_keeps_speaker_roles_disjoint(tmp_path):
    rows = []
    for speaker in range(1, 5):
        for index in range(3):
            path = tmp_path/f"{speaker}-{index}.wav"
            sf.write(path, np.full(2048, .01*speaker+.001*index), 16000)
            rows.append(dict(id=f"{speaker}-{index}", speaker=str(speaker), corpus="librispeech",
                path=str(path), split="dev-clean", role="validation", sample_rate=16000,
                num_samples=2048, start=0, stop=2048))
    manifest = tmp_path/"validation.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    exclusions = tmp_path/"used.csv"
    exclusions.write_text("id\n1-0\n2-0\n3-0\n4-0\n")
    cohort, protocol = speech_examples(manifest, exclusions, samples=1024, sample_rate=16000,
                                       speakers=4, clips_per_speaker=2, seed=0)
    assert len(cohort) == 8
    assert all(r["id"] not in protocol["excluded_ids"] for r in cohort)
    assert not set(protocol["calibration_speakers"]) & set(protocol["measurement_speakers"])
    assert all(r["start"] == 512 and r["stop"] == 1536 for r in cohort)


def test_operating_points_enforce_actual_decoded_limits():
    rows = [dict(id="a", method="l1", nmse=d, bitrate_kbps=r) for d, r in ((.11, 5), (.09, 10), (.01, 30))]
    selected = operating_points(rows, [.1], [9, 20])
    assert next(r for r in selected if r["mode"] == "nmse")["bitrate_kbps"] == 10
    assert next(r for r in selected if r["mode"] == "kbps" and r["limit"] == 9)["nmse"] == .11
    assert next(r for r in selected if r["mode"] == "kbps" and r["limit"] == 20)["nmse"] == .09


def test_small_study_writes_decodable_artifacts_and_reports_failures(tmp_path):
    cfg = StudyConfig("l1_check", "l1", project_root=str(tmp_path), parameters=dict(
        seconds=.064, workers=1, targets=[.1], quantization_bits=[4, 8], max_iterations=1000))
    run = run_study(cfg)
    with (run/"metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20  # Five synthetic inputs, two methods, two precisions.
    assert all((run/r["bitstream"]).is_file() for r in rows)
    assert (run/"report.md").exists() and (run/"rate_distortion.png").exists()
    summary = json.loads((run/"summary.json").read_text())
    assert summary["encoded_points"] == 20 and summary["l1_fits"] == 5
    from tools.export_l1_results import export

    published = tmp_path/"published"
    export(run, published)
    original = (published/"l1_metrics.csv").read_bytes()
    assert json.loads((published/"l1_results.json").read_text())["encoded_points"] == 20
    # A changed source must not silently replace the already published results.
    (run/"metrics.csv").write_text("changed after completion")
    with pytest.raises(ValueError, match="Run artifact changed"):
        export(run, published)
    assert (published/"l1_metrics.csv").read_bytes() == original
