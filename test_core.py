import json
from pathlib import Path
import tempfile
import unittest

from core import Evaluation, REQUIRED_CHECKS, Run, fingerprint


class FakeTrainer:
    def __init__(self):
        self.calls = 0

    def train(self, attempt):
        self.calls += 1
        checkpoint = attempt.directory / "weights.fake"
        checkpoint.write_text(f"checkpoint-{attempt.number}")
        return checkpoint


class FakeReviewer:
    def __init__(self, failed=(), uncertain=False, wrong_suite=False):
        self.failed = failed
        self.uncertain = uncertain
        self.wrong_suite = wrong_suite

    def evaluate(self, checkpoint, attempt):
        evidence = attempt.directory / "review.json"
        checks = {name: name not in self.failed for name in REQUIRED_CHECKS}
        evidence.write_text(json.dumps(checks))
        return Evaluation(
            "wrong" if self.wrong_suite else fingerprint(attempt.suite),
            checks, (evidence,), self.uncertain,
        )


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.suite = root / "evaluation.json"
        self.suite.write_text(json.dumps({"prompts": ["held-out portrait descriptor"]}))
        self.run = Run.create(root / "run", self.suite, "licensed-data-v1", 3)
        self.trainer = FakeTrainer()

    def approve(self):
        self.run.approve_setup("Dataset rights checked; suite and bounded compute approved.")

    def test_setup_is_a_persisted_human_gate(self):
        with self.assertRaises(ValueError):
            self.run.execute(self.trainer, FakeReviewer())
        self.assertEqual(self.trainer.calls, 0)
        self.approve()
        self.assertEqual(Run(self.run.directory).state["status"], "ready")

    def test_approved_candidate_is_promoted_and_reloadable(self):
        self.approve()
        self.run.execute(self.trainer, FakeReviewer())
        state = Run(self.run.directory).state
        self.assertEqual(state["status"], "promoted")
        self.assertEqual(state["incumbent"]["attempt"], 1)
        self.assertEqual(self.trainer.calls, 1)

    def test_failed_checks_exhaust_budget_without_promotion(self):
        self.approve()
        self.run.execute(self.trainer, FakeReviewer(failed=("pixel_quality",)))
        self.assertEqual(self.trainer.calls, 3)
        self.assertIsNone(self.run.state["incumbent"])
        self.assertEqual(self.run.state["reason"], "budget_exhausted")
        with self.assertRaises(ValueError):
            self.run.execute(self.trainer, FakeReviewer())

    def test_uncertain_reviews_retry_automatically_before_human_escalation(self):
        self.approve()
        self.run.execute(self.trainer, FakeReviewer(uncertain=True))
        self.assertEqual(self.trainer.calls, 3)
        self.assertEqual(self.run.state["reason"], "budget_exhausted")
        self.assertIsNone(self.run.state["incumbent"])

    def test_suite_change_blocks_training(self):
        self.approve()
        (self.run.directory / "suite.json").write_text("{}")
        self.run.execute(self.trainer, FakeReviewer())
        self.assertEqual(self.run.state["reason"], "suite_changed")
        self.assertEqual(self.trainer.calls, 0)

    def test_wrong_suite_review_is_a_backend_failure(self):
        self.approve()
        self.run.execute(self.trainer, FakeReviewer(wrong_suite=True))
        self.assertEqual(self.run.state["status"], "failed")
        self.assertEqual(self.trainer.calls, 1)
        self.assertIsNone(self.run.state["incumbent"])

    def test_regression_preserves_incumbent(self):
        self.approve()
        self.run.execute(self.trainer, FakeReviewer())
        incumbent = self.run.state["incumbent"]
        self.run.continue_cycle()
        self.run.execute(self.trainer, FakeReviewer(failed=("non_regression",)))
        self.assertEqual(self.run.state["incumbent"], incumbent)
        self.assertEqual(self.run.state["reason"], "budget_exhausted")

    def test_second_promotion_can_roll_back(self):
        self.approve()
        self.run.execute(self.trainer, FakeReviewer())
        self.run.continue_cycle()
        self.run.execute(self.trainer, FakeReviewer())
        self.assertEqual(self.run.state["incumbent"]["attempt"], 2)
        self.run.rollback("Deployment issue discovered.")
        self.assertEqual(Run(self.run.directory).state["incumbent"]["attempt"], 1)
        self.assertIsNone(self.run.state["previous"])

    def test_interrupted_run_cannot_silently_resume(self):
        self.approve()
        self.run.state["status"] = "running"
        self.run._save()
        with self.assertRaises(ValueError):
            Run(self.run.directory).execute(self.trainer, FakeReviewer())

    def test_external_artifact_is_rejected(self):
        self.approve()
        suite = self.suite

        class ExternalTrainer:
            def train(self, attempt):
                return suite

        self.run.execute(ExternalTrainer(), FakeReviewer())
        self.assertEqual(self.run.state["status"], "failed")
        self.assertIsNone(self.run.state["incumbent"])

    def test_changed_checkpoint_is_rejected(self):
        self.approve()

        class MutatingReviewer(FakeReviewer):
            def evaluate(self, checkpoint, attempt):
                checkpoint.write_text("mutated")
                return super().evaluate(checkpoint, attempt)

        self.run.execute(self.trainer, MutatingReviewer())
        self.assertEqual(self.run.state["status"], "failed")
        self.assertIsNone(self.run.state["incumbent"])

    def test_missing_checks_or_evidence_cannot_pass(self):
        self.approve()

        class IncompleteReviewer:
            def evaluate(self, checkpoint, attempt):
                return Evaluation(fingerprint(attempt.suite), {}, ())

        self.run.execute(self.trainer, IncompleteReviewer())
        self.assertEqual(self.run.state["status"], "failed")
        self.assertIsNone(self.run.state["incumbent"])


if __name__ == "__main__":
    unittest.main()
