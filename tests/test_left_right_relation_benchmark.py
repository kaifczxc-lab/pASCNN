import csv
import json
from pathlib import Path
import shutil
import unittest

from pascnn.training.left_right_relation_benchmark import (
    ALL_OOD_COMBOS,
    LeftRightRelationBenchmarkConfig,
    _split_ood_combo_pool,
    run_left_right_relation_benchmark,
)


def _prepare_output_dir(name: str) -> Path:
    output_dir = Path.cwd() / ".test_artifacts" / name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class LeftRightRelationBenchmarkTests(unittest.TestCase):
    def test_ood_combo_pool_is_split_into_disjoint_seeded_halves(self) -> None:
        val_ood_combos, test_ood_combos = _split_ood_combo_pool(seed=7)
        self.assertGreater(len(val_ood_combos), 0)
        self.assertGreater(len(test_ood_combos), 0)
        self.assertEqual(set(val_ood_combos).intersection(test_ood_combos), set())
        self.assertEqual(
            set(val_ood_combos).union(test_ood_combos),
            set(ALL_OOD_COMBOS),
        )

    def test_quick_relation_benchmark_runs_and_writes_report(self) -> None:
        output_dir = _prepare_output_dir("left_right_relation_benchmark")
        try:
            config = LeftRightRelationBenchmarkConfig(
                output_dir=str(output_dir),
                train_samples=64,
                val_samples=32,
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
            outputs = run_left_right_relation_benchmark(config)
            report_payload = json.loads(Path(outputs.report_path).read_text(encoding="utf-8"))

            self.assertTrue(Path(outputs.report_path).exists())
            self.assertEqual(len(outputs.transformer_run.history), 1)
            self.assertEqual(len(outputs.codebook_run.history), 1)
            self.assertEqual(len(outputs.linear_state_run.history), 1)
            self.assertGreater(outputs.transformer_run.parameter_count, 0)
            self.assertGreater(outputs.codebook_run.parameter_count, 0)
            self.assertGreater(outputs.linear_state_run.parameter_count, 0)
            self.assertIn("transformer_run", report_payload)
            self.assertIn("codebook_run", report_payload)
            self.assertIn("linear_state_run", report_payload)
            self.assertTrue(
                0.0 <= report_payload["transformer_run"]["history"][0]["test_ood_accuracy"] <= 1.0
            )
            self.assertTrue(
                0.0 <= report_payload["codebook_run"]["history"][0]["test_ood_accuracy"] <= 1.0
            )
            self.assertTrue(
                0.0 <= report_payload["linear_state_run"]["history"][0]["test_ood_accuracy"] <= 1.0
            )
            for run_name in ("transformer_run", "codebook_run", "linear_state_run"):
                checkpoint_path = Path(report_payload[run_name]["selected_checkpoint_path"])
                self.assertTrue(checkpoint_path.exists())
                csv_paths = report_payload[run_name]["selected_prediction_csv_paths"]
                self.assertEqual(
                    set(csv_paths.keys()),
                    {"val_ood", "test_iid", "test_ood"},
                )
                for split_name, csv_path in csv_paths.items():
                    path = Path(csv_path)
                    self.assertTrue(path.exists(), msg=f"Missing CSV for {run_name}:{split_name}")
                    with path.open("r", encoding="utf-8", newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertGreater(len(rows), 0)
                    self.assertIn("target_class", rows[0])
                    self.assertIn("predicted_class", rows[0])
                    self.assertIn("confidence", rows[0])
                    self.assertEqual(rows[0]["split"], split_name)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
