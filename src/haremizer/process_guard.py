"""Owned process-group guard shared by local coordinators."""
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

def child_limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_process(stage, argv, timeout, name, *, bounds, save, deadline):
    bounds()
    limit = min(timeout, deadline - time.time())
    if limit <= 0:
        raise ValueError("deadline_exhausted")
    environment = {**os.environ, "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    directory = Path(stage["directory"])
    record = {"argv": argv, "stdout": str(directory / f"{name}.stdout.log"),
              "stderr": str(directory / f"{name}.stderr.log"), "timeout_seconds": limit,
              "status": "running", "code": None, "memory_outcome": "not_reported"}
    stage["processes"].append(record)
    save()
    process = None
    started = time.monotonic()
    try:
        with open(record["stdout"], "xb") as stdout, open(record["stderr"], "xb") as stderr:
            process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=environment,
                                       cwd=directory, start_new_session=True, preexec_fn=child_limits)
            record["owned_pgid"] = process.pid
            save()
            last_storage = time.monotonic()
            while process.poll() is None:
                if time.monotonic() - started >= limit:
                    raise TimeoutError(f"{name} timeout")
                if time.monotonic() - last_storage >= 5:
                    bounds()
                    last_storage = time.monotonic()
                time.sleep(0.1)
            record["code"] = process.returncode
            if process.returncode:
                raise RuntimeError(f"{name} exit {process.returncode}")
            record["status"] = "completed"
    except BaseException as error:
        if process is not None:
            # Only this child's new session/group, never unrelated GPU processes.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                record["cleanup_pending"] = True
            record["code"] = process.returncode
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["duration_seconds"] = time.monotonic() - started
        save()
    bounds()
