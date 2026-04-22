import json
from pathlib import Path
import shutil
import unittest

from pascnn.training.left_right_relation_benchmark import (
    LeftRightRelationBenchmarkConfig,
    run_left_right_relation_benchmark,
)
from pascnn.training.left_right_relation_leakage_audit import (
    run_left_right_relation_leakage_audit,
)


def _prepare_output_dir(name: str) -> Path:
    output_dir = Path.cwd() / ".test_artifacts" / name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class LeftRightRelationLeakageAuditTests(unittest.TestCase):
    def test_quick_leakage_audit_runs_and_writes_report(self) -> None:
        output_dir = _prepare_output_dir("left_right_relation_leakage_audit")
        try:
            benchmark_config = LeftRightRelationBenchmarkConfig(
                output_dir=str(output_dir / "benchmark"),
                train_samples=64,
                val_samples=32,
                val_ood_samples=32,
                test_iid_samples=32,
                test_ood_samples=32,
                batch_size=8,
                eval_batch_size=8,
                epochs=1,
                device="cpu",
                patch_size=4,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                ffn_hidden_dim=32,
                dropout=0.0,
                num_logic_heads=2,
                core_hidden_dim=4,
                core_branch_depth=2,
                core_branch_base=2,
                core_num_layers=1,
            )
            benchmark_outputs = run_left_right_relation_benchmark(benchmark_config)
            audit_outputs = run_left_right_relation_leakage_audit(
                benchmark_report_path=benchmark_outputs.report_path,
            )
            audit_payload = json.loads(Path(audit_outputs.report_path).read_text(encoding="utf-8"))

            self.assertTrue(Path(audit_outputs.report_path).exists())
            self.assertIn("duplicate_hash_counts", audit_payload)
            self.assertIn("transformer_result", audit_payload)
            self.assertTrue(0.0 <= audit_payload["chance_accuracy"] <= 1.0)
            self.assertTrue(
                0.0
                <= audit_payload["transformer_result"]["csv_selected_test_ood_accuracy"]
                <= 1.0
            )
            self.assertTrue(
                0.0
                <= audit_payload["linear_state_result"]["broken_pair_test_ood_accuracy"]
                <= 1.0
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
