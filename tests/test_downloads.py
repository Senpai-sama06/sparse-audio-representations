import io
import tarfile
import zipfile
import pytest
from lrac.data.downloads import safe_extract


def test_zip_path_traversal_is_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="escapes destination"):
        safe_extract(path, tmp_path / "target")
    assert not (tmp_path / "outside.txt").exists()


def test_tar_links_are_rejected(tmp_path):
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as archive:
        item = tarfile.TarInfo("link")
        item.type = tarfile.SYMTYPE
        item.linkname = "../outside"
        archive.addfile(item)
    with pytest.raises(ValueError, match="Unsupported archive entry"):
        safe_extract(path, tmp_path / "target")


def test_regular_tar_is_extracted(tmp_path):
    path = tmp_path / "good.tar"
    with tarfile.open(path, "w") as archive:
        item = tarfile.TarInfo("directory/example.txt")
        item.size = 5
        archive.addfile(item, io.BytesIO(b"hello"))
    safe_extract(path, tmp_path / "target")
    assert (tmp_path / "target/directory/example.txt").read_text() == "hello"
