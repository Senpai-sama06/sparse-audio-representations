"""Explicit, resumable-at-file-level acquisition with checksums and safe extraction."""
from pathlib import Path
import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from lrac.utils import write_json


def fetch(url, target, checksum=None, algorithm="sha256"):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and checksum and digest(target, algorithm) == checksum:
        return target
    temporary = target.with_suffix(target.suffix + ".partial")
    print(f"Downloading {url} -> {target}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    if checksum and digest(temporary, algorithm) != checksum:
        raise ValueError(f"Checksum mismatch: {target}; partial retained for inspection")
    temporary.replace(target)
    return target


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    def check(name):
        path = (destination / name).resolve()
        if not path.is_relative_to(destination):
            raise ValueError(f"Archive entry escapes destination: {name}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                check(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not accepted")
            z.extractall(destination)
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                check(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsupported archive entry: {member.name}")
            # Manual regular-file extraction also supports older Python 3.11
            # builds that do not yet expose tarfile's extraction filters.
            for member in t.getmembers():
                target = destination / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)


def acquire(source, root, librispeech_splits=None):
    root = Path(root)
    if source == "librispeech":
        base = "https://www.openslr.org/resources/12/"
        text = urllib.request.urlopen(base + "md5sum.txt", timeout=60).read().decode()
        checks = {line.split()[-1].lstrip("*"): line.split()[0] for line in text.splitlines() if len(line.split()) == 2}
        names = librispeech_splits if librispeech_splits is not None else ["train-clean-100", "dev-clean", "test-clean", "test-other"]
        archives = [(name + ".tar.gz", base + name + ".tar.gz", checks[name + ".tar.gz"], "md5") for name in names]
    elif source == "vctk":
        archives = [("VCTK-Corpus-0.92.zip", "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip", "f96258be9fdc2cbff6559541aae7ea4f59df3fcaf5cf963aae5ca647357e359c", "sha256")]
    else:
        raise ValueError("Synthetic data does not require a download")
    for name, url, checksum, algorithm in archives:
        marker = root / ".acquisition" / (name + ".json")
        if marker.exists():
            continue
        archive = fetch(url, root / "archives" / name, checksum, algorithm)
        safe_extract(archive, root)
        write_json(marker, {"url": url, "checksum": checksum, "algorithm": algorithm})
