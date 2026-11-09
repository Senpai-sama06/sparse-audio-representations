"""Lightweight process identity, advisory locks, and atomic progress records."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
import uuid


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Windows readers/virus scanners can briefly deny replacement of an open
        # destination. Keep the complete old snapshot until replacement succeeds.
        for attempt in range(25):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 24:
                    raise
                time.sleep(min(.01 * (attempt + 1), .1))
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    for attempt in range(5):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return default
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.02)


def process_identity(pid=None):
    """Return a birth token as well as PID, so recycled PIDs are not mistaken for jobs."""
    pid = os.getpid() if pid is None else pid
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(wintypes.HANDLE(handle), ctypes.byref(exit_code)) or exit_code.value != 259:
                return None
            creation, end, system, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(wintypes.HANDLE(handle), ctypes.byref(creation), ctypes.byref(end),
                                          ctypes.byref(system), ctypes.byref(user)):
                return None
            token = str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
        finally:
            kernel.CloseHandle(wintypes.HANDLE(handle))
    else:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                return None
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            token = boot + ":" + fields[19]
        except (FileNotFoundError, ProcessLookupError):
            return None
    return dict(pid=pid, birth=token, host=socket.gethostname())


def process_alive(identity):
    if not identity or identity.get("host") != socket.gethostname():
        return False
    return process_identity(identity["pid"]) == identity


@contextmanager
def file_lock(path, timeout=86400):
    """OS-released lock: a crashed process cannot leave a permanently stale lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(path, "a+b")
    if path.stat().st_size == 0:
        stream.write(b"0")
        stream.flush()
    started, announced = time.monotonic(), False
    acquired = False
    try:
        while True:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f"Another process holds {path}")
                if not announced:
                    print(f"Waiting for shared setup/run lock: {path}", flush=True)
                    announced = True
                time.sleep(.2)
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


class Progress:
    def __init__(self, output, method, interval=5):
        self.path = Path(output) / "progress.json"
        self.started = time.monotonic()
        self.written = 0.0
        self.interval = interval
        self.state = dict(method=method, process=process_identity(), started_at=utc_now(),
                          phase="starting", epoch=0, completed_epochs=0, best_epoch=0,
                          best_value=None, batch=0, batches=0)
        self.update()

    def update(self, **fields):
        self.state.update(fields)
        self.state.update(updated_at=utc_now(), elapsed_seconds=time.monotonic() - self.started)
        atomic_json(self.path, self.state)
        self.written = time.monotonic()

    def pulse(self, **fields):
        self.state.update(fields)
        if time.monotonic() - self.written >= self.interval:
            self.update()
