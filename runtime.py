"""Read-only runtime facts; optional bounded GPU primitive verification, never training."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile


# Run in a child so a driver hang cannot hold the preparation process indefinitely.
TORCH_PROBE = r'''
import json
import torch
result = {"torch_version": torch.__version__, "hip_version": torch.version.hip,
          "cuda_version": torch.version.cuda, "gpu_visible": torch.cuda.is_available(),
          "gpu_operations_verified": False}
try:
    if not result["gpu_visible"]:
        raise RuntimeError("torch reports no GPU; no CPU fallback")
    result["device"] = torch.cuda.get_device_name(0)
    with torch.no_grad():
        x = torch.ones((16, 16), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        if not torch.all(y == 16).item():
            raise RuntimeError("GPU matrix result mismatch")
    x = torch.ones((16, 16), device="cuda", requires_grad=True)
    (x.square().sum()).backward()
    torch.cuda.synchronize()
    if not torch.all(x.grad == 2).item():
        raise RuntimeError("GPU backward result mismatch")
    result["gpu_operations_verified"] = True
except Exception as error:
    result["error"] = f"{type(error).__name__}: {error}"
print(json.dumps(result))
'''


def bounded_command(command: list[str], timeout: int = 10) -> dict:
    """Fixed callers only; cap elapsed time and returned diagnostic text."""
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr,
                                       timeout=timeout, check=False)
            result = {"command": command, "returncode": completed.returncode}
        except (OSError, subprocess.TimeoutExpired) as error:
            result = {"command": command, "error": f"{type(error).__name__}: {error}"}
        for name, stream in (("stdout", stdout), ("stderr", stderr)):
            stream.seek(0)
            text = stream.read(16385)
            result[name] = text[:16384].decode("utf-8", errors="replace")
            result[f"{name}_truncated"] = len(text) > 16384
        return result


def probe_runtime(torch_probe: bool = False) -> dict:
    facts = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(),
        "environment": {key: os.environ.get(key) for key in (
            "HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES")},
        "management": [], "gpu_operations_verified": False,
        "probe_mode": "explicit_environment_override" if os.environ.get("HSA_OVERRIDE_GFX_VERSION") else "native",
        "compatibility_workaround_requires_human_acknowledgment": bool(os.environ.get("HSA_OVERRIDE_GFX_VERSION")),
        "torch_probe": {"status": "not_requested"},
        "scope": "visibility and small tensor primitives only; not model compatibility or training",
    }
    for name, arguments in (("lspci", ["-nn"]), ("nvidia-smi", ["-L"]),
                            ("rocm-smi", ["--showproductname"])):
        executable = shutil.which(name)
        facts["management"].append(
            bounded_command([executable, *arguments]) if executable else
            {"command": name, "error": "not installed"})
    if torch_probe:
        child = bounded_command([sys.executable, "-B", "-c", TORCH_PROBE], timeout=30)
        facts["torch_probe"] = child
        child["status"] = "blocked"
        if child.get("returncode", 0) < 0:
            child["signal"] = -child["returncode"]
        if child.get("returncode") == 0 and not child["stdout_truncated"]:
            try:
                evidence = json.loads(child["stdout"])
                facts["torch_probe"]["evidence"] = evidence
                facts["gpu_operations_verified"] = evidence.get("gpu_operations_verified") is True
                if facts["gpu_operations_verified"]:
                    child["status"] = "verified"
            except (ValueError, AttributeError):
                child["error"] = "probe did not return a JSON object"
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-probe", action="store_true",
                        help="opt in to a 30-second GPU math/inference/backward probe")
    args = parser.parse_args()
    print(json.dumps(probe_runtime(args.torch_probe), indent=2))


if __name__ == "__main__":
    main()
