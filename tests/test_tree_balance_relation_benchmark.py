import json
from pathlib import Path
import shutil
import unittest

from pascnn.training.tree_relation_benchmark import TreeRelationBenchmarkConfig
from pascnn.training.tree_balance_relation_benchmark import (
    TREE_BALANCE_STATE_NAMES,
    TREE_BALANCE_RELATION_CLASS_NAMES,
    build_tree_balance_relation_datasets,
    run_tree_balance_relation_benchmark,
)


def _prepare_output_dir(name: str) -> Path:
    output_dir = Path.cwd() / ".test_artifacts" / name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TreeBalanceRelationBenchmarkTests(unittest.TestCase):
    def test_tree_balance_relation_dataset_respects_depth_splits_and_labels(self) -> None:
        config = TreeRelationBenchmarkConfig(
            output_dir="unused",
            train_samples_per_class=4,
            val_iid_samples_per_class=3,
            val_ood_samples_per_class=3,
            test_iid_samples_per_class=3,
            test_ood_samples_per_class=3,
            train_depths=(2, 3),
            ood_depths=(4, 5),
            sequence_length=256,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            ffn_hidden_dim=32,
            num_logic_heads=2,
            core_hidden_dim=4,
            core_branch_depth=2,
            core_branch_base=2,
            core_num_layers=1,
        )
        bundle = build_tree_balance_relation_datasets(config)

        for metadata in bundle.train_dataset.metadata + bundle.val_iid_dataset.metadata + bundle.test_iid_dataset.metadata:
            self.assertIn(metadata.pair_depth, config.train_depths)
            self.assertEqual(metadata.left_state_name, TREE_BALANCE_STATE_NAMES[metadata.left_state])
            self.assertEqual(metadata.right_state_name, TREE_BALANCE_STATE_NAMES[metadata.right_state])
            expected_name = TREE_BALANCE_RELATION_CLASS_NAMES[metadata.target_class]
            self.assertEqual(metadata.target_name, expected_name)
            if metadata.target_class == 0:
                self.assertLess(metadata.left_state, metadata.right_state)
            elif metadata.target_class == 1:
                self.assertEqual(metadata.left_state, metadata.right_state)
            else:
                self.assertGreater(metadata.left_state, metadata.right_state)

        for metadata in bundle.val_ood_dataset.metadata + bundle.test_ood_dataset.metadata:
            self.assertIn(metadata.pair_depth, config.ood_depths)

    def test_quick_tree_balance_relation_benchmark_runs_and_writes_report(self) -> None:
        output_dir = _prepare_output_dir("tree_balance_relation_benchmark")
        try:
            config = TreeRelationBenchmarkConfig(
                output_dir=str(output_dir),
                train_samples_per_class=12,
                val_iid_samples_per_class=6,
                val_ood_samples_per_class=6,
                test_iid_samples_per_class=6,
                test_ood_samples_per_class=6,
                batch_size=6,
                eval_batch_size=6,
                epochs=1,
                device="cpu",
                train_depths=(2, 3),
                ood_depths=(4, 5),
                sequence_length=256,
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
            outputs = run_tree_balance_relation_benchmark(config)
            report_payload = json.loads(Path(outputs.report_path).read_text(encoding="utf-8"))
            splits_payload = json.loads(Path(outputs.splits_path).read_text(encoding="utf-8"))

            self.assertTrue(Path(outputs.report_path).exists())
            self.assertTrue(Path(outputs.splits_path).exists())
            self.assertEqual(len(outputs.transformer_run.history), 1)
            self.assertEqual(len(outputs.codebook_run.history), 1)
            self.assertEqual(len(outputs.linear_state_run.history), 1)
            self.assertIn("transformer_run", report_payload)
            self.assertIn("codebook_run", report_payload)
            self.assertIn("linear_state_run", report_payload)
            self.assertIn("train", splits_payload)
            self.assertIn("test_ood", splits_payload)
            self.assertEqual(len(splits_payload["train"]), 3 * config.train_samples_per_class)
            self.assertEqual(len(splits_payload["test_ood"]), 3 * config.test_ood_samples_per_class)
            for run_name in ("transformer_run", "codebook_run", "linear_state_run"):
                history = report_payload[run_name]["history"]
                self.assertEqual(len(history), 1)
                self.assertTrue(0.0 <= history[0]["test_iid_accuracy"] <= 1.0)
                self.assertTrue(0.0 <= history[0]["test_ood_accuracy"] <= 1.0)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
