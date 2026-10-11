"""GPU-independent experiment orchestration; training and review are injected."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Protocol


REQUIRED_CHECKS = (
    "format", "pixel_quality", "prompt_adherence", "character_diversity", "non_regression",
)


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Attempt:
    directory: Path
    number: int
    suite: Path
    dataset_id: str
    incumbent: dict | None


@dataclass(frozen=True)
class Evaluation:
    suite_sha256: str
    checks: dict[str, bool]
    evidence: tuple[Path, ...]
    uncertain: bool = False


class Trainer(Protocol):
    def train(self, attempt: Attempt) -> Path:
        """Return a candidate checkpoint inside the attempt directory."""
        ...


class Reviewer(Protocol):
    def evaluate(self, checkpoint: Path, attempt: Attempt) -> Evaluation:
        """Generate held-out samples and assess them independently of training."""
        ...


class Run:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.path = self.directory / "run.json"
        self.state = json.loads(self.path.read_text())

    @classmethod
    def create(cls, directory: Path, suite: Path, dataset_id: str, max_attempts: int = 3) -> Run:
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if not dataset_id.strip():
            raise ValueError("dataset_id is required")
        suite_bytes = Path(suite).read_bytes()
        json.loads(suite_bytes)
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "suite.json").write_bytes(suite_bytes)
        run = cls.__new__(cls)
        run.directory = directory
        run.path = directory / "run.json"
        run.state = {
            "version": 1, "status": "needs_human", "reason": "setup_approval",
            "dataset_id": dataset_id, "suite_sha256": fingerprint(directory / "suite.json"),
            "max_attempts": max_attempts, "attempts": 0,
            "incumbent": None, "previous": None, "events": [],
        }
        run._save()
        return run

    def _save(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(self.state, stream, indent=2)
            stream.write("\n")
        temporary.replace(self.path)

    def _event(self, kind: str, **details) -> None:
        self.state["events"].append({"kind": kind, **details})
        self._save()

    def approve_setup(self, note: str) -> None:
        if self.state["reason"] != "setup_approval" or self.state["status"] != "needs_human":
            raise ValueError("run is not awaiting setup approval")
        if not note.strip():
            raise ValueError("record dataset rights, evaluation policy, and compute approval")
        self.state.update(status="ready", reason=None)
        self._event("setup_approved", note=note)

    def _artifact(self, path: Path, directory: Path) -> Path:
        path = Path(path).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("artifact must be a file inside the attempt directory")
        return path

    def execute(self, trainer: Trainer, reviewer: Reviewer) -> dict:
        """Run serial attempts; stop at promotion, error, or budget."""
        if self.state["status"] != "ready":
            raise ValueError("run must be ready; no automatic resume or approval")
        suite = self.directory / "suite.json"
        if fingerprint(suite) != self.state["suite_sha256"]:
            self.state.update(status="failed", reason="suite_changed")
            self._event("suite_changed")
            return self.state
        while self.state["attempts"] < self.state["max_attempts"]:
            self.state["attempts"] += 1
            number = self.state["attempts"]
            directory = self.directory / f"attempt-{number:03}"
            self.state.update(status="running", reason=None)
            self._event("attempt_started", number=number)
            try:
                directory.mkdir()
                attempt = Attempt(directory, number, suite, self.state["dataset_id"], self.state["incumbent"])
                if attempt.incumbent and fingerprint(Path(attempt.incumbent["checkpoint"])) != attempt.incumbent["sha256"]:
                    raise ValueError("incumbent checkpoint changed")
                checkpoint = self._artifact(trainer.train(attempt), directory)
                digest = fingerprint(checkpoint)
                report = reviewer.evaluate(checkpoint, attempt)
                if fingerprint(suite) != self.state["suite_sha256"] or report.suite_sha256 != self.state["suite_sha256"]:
                    raise ValueError("evaluation suite mismatch")
                if fingerprint(checkpoint) != digest:
                    raise ValueError("checkpoint changed during evaluation")
                if set(report.checks) != set(REQUIRED_CHECKS) or any(type(v) is not bool for v in report.checks.values()):
                    raise ValueError("review must supply all required boolean checks")
                if type(report.uncertain) is not bool or not report.evidence:
                    raise ValueError("review must supply evidence and a boolean uncertainty flag")
                evidence = [str(self._artifact(path, directory)) for path in report.evidence]
                candidate = {"checkpoint": str(checkpoint), "sha256": digest,
                             "attempt": number, "checks": report.checks, "evidence": evidence}
                self._event("evaluated", candidate=candidate, uncertain=report.uncertain)
                if not all(report.checks.values()):
                    self._event("rejected", number=number)
                    continue
                if report.uncertain:
                    self._event("uncertain_rejected", number=number)
                    continue
                self.state["previous"] = self.state["incumbent"]
                self.state.update(incumbent=candidate, status="promoted", reason=None)
                self._event("promoted", number=number)
                return self.state
            except Exception as error:
                self.state.update(status="failed", reason="backend_error")
                self._event("backend_error", number=number, error=f"{type(error).__name__}: {error}")
                return self.state
        self.state.update(status="needs_human", reason="budget_exhausted")
        self._event("budget_exhausted")
        return self.state

    def continue_cycle(self) -> None:
        if self.state["status"] != "promoted":
            raise ValueError("only a promoted run can continue")
        self.state.update(status="ready", reason=None)
        self._event("cycle_requested")

    def rollback(self, note: str) -> None:
        if self.state["status"] != "promoted" or not self.state["previous"]:
            raise ValueError("no previous promoted checkpoint is available")
        if not note.strip():
            raise ValueError("a rollback reason is required")
        previous = self.state["previous"]
        if fingerprint(Path(previous["checkpoint"])) != previous["sha256"]:
            raise ValueError("previous checkpoint changed")
        self.state.update(incumbent=previous, previous=None)
        self._event("rolled_back", note=note)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("directory", type=Path)
    init.add_argument("--suite", type=Path, required=True)
    init.add_argument("--dataset-id", required=True)
    init.add_argument("--max-attempts", type=int, default=3)
    status = commands.add_parser("status")
    status.add_argument("directory", type=Path)
    approve = commands.add_parser("approve-setup")
    approve.add_argument("directory", type=Path)
    approve.add_argument("--note", required=True)
    args = parser.parse_args()
    if args.command == "init":
        run = Run.create(args.directory, args.suite, args.dataset_id, args.max_attempts)
    else:
        run = Run(args.directory)
        if args.command == "approve-setup":
            run.approve_setup(args.note)
    print(json.dumps(run.state, indent=2))


if __name__ == "__main__":
    main()
