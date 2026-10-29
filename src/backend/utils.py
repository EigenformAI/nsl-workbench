import subprocess
import urllib.request

from loguru import logger


def is_http_ready(url: str, timeout_sec: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def get_recent_log_lines(log_path: str, n: int = 15) -> str:
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            return "".join(lines[-n:])
    except Exception as e:
        return f"(could not read log: {e})"


def terminate_process(process: subprocess.Popen, name: str) -> None:
    if process.poll() is not None:
        return

    logger.info(f"Stopping {name} process (pid={process.pid})...")
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning(f"{name} did not stop in time. Killing process {process.pid}...")
        process.kill()
        process.wait(timeout=5)
