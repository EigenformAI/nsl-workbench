import time
from contextlib import contextmanager
from datetime import datetime
import hashlib
import random
import re
import string
from typing import (
    Callable,
)

import ctypes
import subprocess
import threading


_SAFE_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def nanoid(size=21) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(size))


def _raise_in_thread(thread_id: int, exc_type: type) -> bool:
    """Raise *exc_type* asynchronously in the thread identified by *thread_id*.

    Uses ``ctypes.pythonapi.PyThreadState_SetAsyncExc`` which is CPython-specific
    but works from any thread (not just main) and interrupts ``time.sleep`` and
    other pure-Python blocking calls.

    Returns True if the exception was successfully scheduled, False otherwise.
    """
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id),
        ctypes.py_object(exc_type),
    )
    return ret == 1


@contextmanager
def timeout(seconds: float, callback: Callable = lambda: None):
    """Context manager that raises ``TimeoutError`` in the *calling* thread
    if the body takes longer than *seconds*.

    Uses ``PyThreadState_SetAsyncExc`` to inject the exception into the correct
    thread (works from worker threads, not just main).  A ``threading.Timer``
    fires the injection after the deadline.

    Args:
        seconds: Maximum wall-clock seconds before the timeout fires.
        callback: Optional callable invoked (in the timer thread) just before
            the exception is raised.

    Raises:
        TimeoutError: If the code execution exceeds *seconds*.
    """
    target_thread_id = threading.current_thread().ident
    assert target_thread_id is not None
    timer = None

    def _on_timeout() -> None:
        callback()
        _raise_in_thread(target_thread_id, TimeoutError)

    timer = threading.Timer(seconds, _on_timeout)
    timer.daemon = True
    timer.start()

    try:
        yield
    finally:
        if timer:
            timer.cancel()


def unflatten_toml_dict(d: dict) -> dict:
    result = {}
    for key, value in d.items():
        parts = key.split(".")
        current_level = result
        for i, part in enumerate(parts):
            if i == len(parts) - 1:  # Last part, assign value
                current_level[part] = value
            else:
                current_level = current_level.setdefault(part, {})

    return result


def generate_readable_run_id(
    random_length: int = 6, date_format_str: str = "%Y%m%d", separator: str = "-"
) -> str:
    current_date = datetime.now()

    date_part = current_date.strftime(date_format_str)

    characters = string.ascii_lowercase + string.digits
    random_part = "".join(random.choices(characters, k=random_length))

    # 4. Combine the parts
    run_id = f"{date_part}{separator}{random_part}"

    return run_id


def validate_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or run_id in {".", ".."}
        or _SAFE_RUN_ID_PATTERN.fullmatch(run_id) is None
    ):
        raise ValueError(
            "run_id must be a non-empty safe path component containing only "
            "ASCII letters, digits, '.', '_', or '-'"
        )
    return run_id


def get_formatted_repo_info():
    """
    Gets current Git repository information formatted as:
    "{commit-hash-capitalized}-{branch-capitalized}-{number-of-uncommitted-files}"

    If not in a git repository, returns a default identifier based on timestamp.
    """
    try:
        # 1. Preliminary check: Is this a Git repository?
        subprocess.check_output(
            ["git", "rev-parse", "--git-dir"],
            stderr=subprocess.DEVNULL,
            text=True,
        )

        # 2. Get current commit hash.
        commit_hash_raw = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
        commit_hash_lower = commit_hash_raw.lower()

        # 3. Get current branch name.
        branch_name_raw = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        branch_name_lower = branch_name_raw.lower()

        # 4. Get hash of uncommitted files.
        status_output = subprocess.check_output(
            ["git", "diff", "--stat"], text=True
        ).strip()

        uncommitted_files_hash = "0"
        if status_output:
            uncommitted_files_hash = string_hash(status_output)

        return (
            f"{commit_hash_lower[:6]}-{branch_name_lower}-{uncommitted_files_hash[:6]}"
        )

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repository or git command failed - return default identifier
        timestamp = str(int(time.time()))
        return f"nogit-{timestamp[-6:]}-000000"


def string_hash(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()
