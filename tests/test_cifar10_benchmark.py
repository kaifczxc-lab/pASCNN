import json
from pathlib import Path
import shutil
import unittest

import torch
from torch.utils.data import TensorDataset

from pascnn.training.cifar10_benchmark import CIFAR10BenchmarkConfig
from pascnn.training.cifar10_benchmark import CIFAR10DatasetBundle
from pascnn.training.cifar10_benchmark import run_cifar10_benchmark


def _prepare_output_dir(name: str) -> Path:
    output_dir = Path.cwd() / ".test_artifacts" / name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _make_dataset_bundle() -> CIFAR10DatasetBundle:
    images = torch.zeros(20, 3, 32, 32, dtype=torch.float32)
    labels = torch.arange(20, dtype=torch.long) % 10
    for index in range(20):
        row = (index % 5) * 6
        col = ((index // 5) % 4) * 6
        channel = index % 3
        images[index, channel, row : row + 8, col : col + 8] = 1.0
    train_images = images.repeat((2, 1, 1, 1))
    train_labels = labels.repeat(2)
    return CIFAR10DatasetBundle(
        train_dataset=TensorDataset(train_images, train_labels),
        val_dataset=TensorDataset(images, labels),
        test_dataset=TensorDataset(images, labels),
    )


class CIFAR10BenchmarkTests(unittest.TestCase):
    def test_quick_benchmark_runs_and_writes_report(self) -> None:
        output_dir = _prepare_output_dir("cifar10_benchmark")
        try:
            config = CIFAR10BenchmarkConfig(
                data_root=str(output_dir),
                output_dir=str(output_dir),
                batch_size=8,
                eval_batch_size=8,
                epochs=1,
                seed=7,
                device="cpu",
                patch_size=4,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                ffn_hidden_dim=32,
                dropout=0.0,
                num_logic_heads=5,
                core_hidden_dim=4,
                core_branch_depth=2,
                core_branch_base=2,
                core_num_layers=1,
            )
            outputs = run_cifar10_benchmark(
                config,
                datasets_bundle=_make_dataset_bundle(),
            )
            report_payload = json.loads(Path(outputs.report_path).read_text(encoding="utf-8"))

            self.assertTrue(Path(outputs.report_path).exists())
            self.assertEqual(len(outputs.transformer_run.history), 1)
            self.assertEqual(len(outputs.codebook_run.history), 1)
            self.assertEqual(len(outputs.linear_state_run.history), 1)
            self.assertGreater(outputs.transformer_run.parameter_count, 0)
            self.assertGreater(outputs.codebook_run.parameter_count, 0)
            self.assertGreater(outputs.linear_state_run.parameter_count, 0)
            self.assertEqual(outputs.transformer_run.name, "transformer_image_classifier")
            self.assertEqual(outputs.codebook_run.name, "pascnn_codebook_classifier")
            self.assertEqual(outputs.linear_state_run.name, "pascnn_linear_state_classifier")
            self.assertIn("transformer_run", report_payload)
            self.assertIn("codebook_run", report_payload)
            self.assertIn("linear_state_run", report_payload)
            self.assertTrue(
                0.0 <= report_payload["transformer_run"]["history"][0]["test_accuracy"] <= 1.0
            )
            self.assertTrue(
                0.0 <= report_payload["codebook_run"]["history"][0]["test_accuracy"] <= 1.0
            )
            self.assertTrue(
                0.0 <= report_payload["linear_state_run"]["history"][0]["test_accuracy"] <= 1.0
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
