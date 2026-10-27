from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
import soundfile as sf
from lrac.data.manifests import configured_manifests, read_manifest
from lrac.utils import write_json, file_hash


class AuditError(ValueError):
    pass


def audit(cfg, output=None):
    errors, summaries = [], {}
    speakers, sources, hashes, synthetic_seeds = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    file_cache = {}
    official = {"train-clean-100": "train", "train-clean-360": "train", "train-other-500": "train", "dev-clean": "validation", "test-clean": "test", "test-other": "test"}
    manifests = configured_manifests(cfg)
    if cfg.data.source == "vctk" and cfg.data.vctk_mode == "external":
        manifests = {"test:external": cfg.manifest("external.jsonl")}
    for label, path in manifests.items():
        if not path.exists():
            errors.append(f"Missing manifest: {path}")
            continue
        role = label.split(":")[0]
        rows = read_manifest(path)
        durations = defaultdict(float)
        ids, local_seeds = set(), set()
        if not rows:
            errors.append(f"Empty manifest: {path}")
        for row in rows:
            if row["id"] in ids:
                errors.append(f"Duplicate ID inside {label}: {row['id']}")
            ids.add(row["id"])
            if row.get("role") != role:
                errors.append(f"Role mismatch in {label}: {row['id']}")
            if row["corpus"] == "librispeech" and official.get(row["split"]) != role:
                errors.append(f"Official partition mismatch: {row['id']}")
            if row["corpus"] == "librispeech" and role == "train" and row["split"] != cfg.data.librispeech_train_split:
                errors.append(f"Training partition differs from configuration: {row['id']}")
            if row.get("speaker") is not None:
                key = (row["corpus"], str(row["speaker"]))
                speakers[role].add(key)
                durations[str(row["speaker"])] += (row.get("stop", row["num_samples"]) - row.get("start", 0)) / row["sample_rate"]
            if row["corpus"] == "synthetic":
                if row["seed"] in local_seeds:
                    errors.append(f"Repeated synthetic seed in {label}: {row['seed']}")
                local_seeds.add(row["seed"])
                synthetic_seeds[role].add(int(row["seed"]))
                continue
            source = (row["corpus"], row.get("utterance", row["id"]))
            sources[role].add(source)
            start, stop = row.get("start", 0), row.get("stop", row["num_samples"])
            if not 0 <= start < stop <= row["num_samples"]:
                errors.append(f"Invalid source interval: {row['id']}")
            audio_path = Path(row["path"])
            if not audio_path.exists():
                errors.append(f"Missing audio: {audio_path}")
                continue
            info = sf.info(audio_path)
            if info.frames != row["num_samples"] or info.samplerate != row["sample_rate"] or info.channels != 1:
                errors.append(f"Audio metadata mismatch: {row['id']}")
            if row["corpus"] == "librispeech":
                identifiers = audio_path.stem.split("-")
                if len(identifiers) == 3 and all(v.isdigit() for v in identifiers):
                    if identifiers[0] != str(row["speaker"]) or identifiers[1] != str(row["chapter"]) or audio_path.stem != row["utterance"]:
                        errors.append(f"LibriSpeech filename/manifest mismatch: {row['id']}")
                    if row["split"] not in audio_path.parts:
                        errors.append(f"LibriSpeech source directory mismatch: {row['id']}")
            if cfg.data.hash_audio:
                key = str(audio_path.resolve())
                if key not in file_cache:
                    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
                    h = hashlib.sha256(np.asarray([sr, audio.shape[1]], dtype="<i8").tobytes())
                    h.update(np.asarray(audio, dtype="<f4").tobytes())
                    file_cache[key] = h.hexdigest()
                hashes[role].add(file_cache[key])
        expected = cfg.data.expected_speakers.get(label)
        count = len({(r["corpus"], r["speaker"]) for r in rows if r.get("speaker") is not None})
        if expected is not None and count != expected:
            errors.append(f"{label}: expected {expected} speakers, found {count}")
        summaries[label] = {"records": len(rows), "speakers": count, "seconds": sum((r.get("stop", r["num_samples"]) - r.get("start", 0)) / r["sample_rate"] for r in rows), "seconds_per_speaker": dict(durations), "manifest_sha256": file_hash(path)}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        for title, sets in [("speaker", speakers), ("source recording", sources), ("decoded audio", hashes), ("synthetic seed", synthetic_seeds)]:
            overlap = sets[a] & sets[b]
            if overlap:
                errors.append(f"{title} overlap between {a} and {b}: {len(overlap)}")
    report = {"passed": not errors, "audio_hashing": cfg.data.hash_audio, "errors": errors, "manifests": summaries}
    if output:
        write_json(output, report)
    if errors:
        raise AuditError("Dataset audit failed:\n" + "\n".join(errors[:25]))
    return report
