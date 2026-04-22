import unittest

import torch

from pascnn.adapters.mnist import MNISTDecodeOutputs
from pascnn.diagnostics import collect_core_diagnostics
from pascnn.diagnostics import collect_mnist_diagnostics
from pascnn.models.mnist import MNISTModelOutputs
from pascnn.models.mnist import MNISTReferenceModel
from pascnn.types import CellConfig
from pascnn.types import CellForwardOutputs
from pascnn.types import EdgeCoherenceOutputs
from pascnn.types import LogicalReadoutOutputs
from pascnn.types import PrefixMatchOutputs
from pascnn.types import RestrictDefectOutputs


class PASCNNDiagnosticsTests(unittest.TestCase):
    def test_manual_core_metrics_match_formulas(self) -> None:
        digit_probabilities = torch.tensor(
            [
                [
                    [[0.5, 0.5], [1.0, 0.0]],
                    [[0.25, 0.75], [0.2, 0.8]],
                    [[0.1, 0.9], [0.6, 0.4]],
                    [[0.3, 0.7], [0.9, 0.1]],
                    [[0.8, 0.2], [0.4, 0.6]],
                ]
            ],
            dtype=torch.float32,
        )
        defect = torch.tensor(
            [
                [
                    [3.0 + 4.0j, 0.0 + 0.0j],
                    [0.0 + 2.0j, 0.0 + 0.0j],
                    [1.0 + 0.0j, 2.0 + 0.0j],
                    [4.0 + 0.0j, 3.0 + 0.0j],
                    [0.0 + 1.0j, 0.0 + 1.0j],
                    [1.0 + 1.0j, 1.0 + 1.0j],
                    [2.0 + 0.0j, 0.0 + 2.0j],
                    [3.0 + 0.0j, 4.0 + 0.0j],
                    [1.0 + 2.0j, 2.0 + 1.0j],
                    [0.0 + 0.0j, 5.0 + 0.0j],
                ]
            ],
            dtype=torch.complex64,
        )
        defect_norm_sq = defect.abs().square().sum(dim=-1)
        edge_amplitude = torch.tensor([[0.2, 0.3, 0.4, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        edge_phase = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
        prefix_depth = torch.tensor([[1.0, 1.5, 2.0, 2.5, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25]])
        logic_probabilities = torch.tensor(
            [
                [
                    [0.7, 0.2, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.1, 0.3, 0.6],
                    [0.8, 0.1, 0.1],
                    [0.25, 0.25, 0.5],
                ]
            ],
            dtype=torch.float32,
        )
        cell_outputs = CellForwardOutputs(
            vertex_state=torch.zeros(1, 5, 2, dtype=torch.complex64),
            digit_logits=torch.zeros(1, 5, 2, 2, dtype=torch.float32),
            digit_probabilities=digit_probabilities,
            prefix=PrefixMatchOutputs(
                position_agreement=torch.zeros(1, 10, 2, dtype=torch.float32),
                prefix_compatibility=torch.zeros(1, 10, 2, dtype=torch.float32),
                prefix_depth=prefix_depth,
            ),
            restriction=RestrictDefectOutputs(
                source_transport=torch.zeros(1, 10, 2, dtype=torch.complex64),
                target_transport=torch.zeros(1, 10, 2, dtype=torch.complex64),
                defect=defect,
                defect_norm_sq=defect_norm_sq,
            ),
            coherence=EdgeCoherenceOutputs(
                edge_mean=torch.zeros(1, 10, 2, dtype=torch.complex64),
                edge_amplitude=edge_amplitude,
                edge_phase=edge_phase,
                edge_coefficient=torch.zeros(1, 10, dtype=torch.complex64),
                edge_message=torch.zeros(1, 10, 2, dtype=torch.complex64),
            ),
            readout=LogicalReadoutOutputs(
                logic_amplitudes=torch.zeros(1, 5, 3, dtype=torch.complex64),
                logic_probabilities=logic_probabilities,
            ),
            diagnostics={},
        )

        diagnostics = collect_core_diagnostics(cell_outputs)
        expected_defect_abs_mean = defect.abs().mean(dim=-1)
        expected_code_entropy = -(
            digit_probabilities.clamp_min(torch.finfo(torch.float32).tiny)
            * digit_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1)
        expected_logic_entropy = -(
            logic_probabilities * logic_probabilities.log()
        ).sum(dim=-1)
        expected_code_symbol_usage = digit_probabilities.mean(dim=(0, 1))
        expected_code_symbol_balance_sq = (
            expected_code_symbol_usage - torch.full((2, 2), 0.5, dtype=torch.float32)
        ).square()
        expected_code_symbol_balance_per_depth = expected_code_symbol_balance_sq.mean(dim=-1)
        expected_code_symbol_max_usage_per_depth = expected_code_symbol_usage.max(dim=-1).values
        expected_uniform_prefix_depth_baseline = torch.full_like(prefix_depth, 0.75)
        expected_prefix_depth_normalized = prefix_depth / 2.0
        expected_prefix_depth_lift_over_uniform = (
            prefix_depth - expected_uniform_prefix_depth_baseline
        )
        expected_prefix_depth_to_uniform_ratio = (
            prefix_depth / expected_uniform_prefix_depth_baseline
        )
        expected_defect_norm_sq_normalized = defect_norm_sq / 2.0
        expected_saturation_95 = (edge_amplitude >= 0.95).to(dtype=edge_amplitude.dtype)
        expected_saturation_99 = (edge_amplitude >= 0.99).to(dtype=edge_amplitude.dtype)

        self.assertTrue(
            torch.allclose(diagnostics.edge_defect_abs_mean, expected_defect_abs_mean, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.edge_prefix_depth_normalized,
                expected_prefix_depth_normalized,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.edge_prefix_depth_uniform_baseline,
                expected_uniform_prefix_depth_baseline,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.edge_prefix_depth_lift_over_uniform,
                expected_prefix_depth_lift_over_uniform,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.edge_prefix_depth_to_uniform_ratio,
                expected_prefix_depth_to_uniform_ratio,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(diagnostics.edge_defect_norm_sq, defect_norm_sq, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.edge_defect_norm_sq_normalized,
                expected_defect_norm_sq_normalized,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.edge_coherence_gate_saturation_95,
                expected_saturation_95,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.edge_coherence_gate_saturation_99,
                expected_saturation_99,
                atol=1e-6,
            )
        )
        self.assertTrue(torch.allclose(diagnostics.code_entropy, expected_code_entropy, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                diagnostics.code_symbol_usage,
                expected_code_symbol_usage,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.code_symbol_balance_sq,
                expected_code_symbol_balance_sq,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.code_symbol_balance_per_depth,
                expected_code_symbol_balance_per_depth,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.code_symbol_max_usage_per_depth,
                expected_code_symbol_max_usage_per_depth,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.logic_probability_entropy,
                expected_logic_entropy,
                atol=1e-6,
            )
        )
        self.assertTrue(torch.allclose(diagnostics.wave_prefix_depth, prefix_depth[:, 3], atol=1e-6))
        self.assertTrue(
            torch.allclose(
                diagnostics.wave_prefix_depth_normalized,
                expected_prefix_depth_normalized[:, 3],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.wave_prefix_depth_uniform_baseline,
                expected_uniform_prefix_depth_baseline[:, 3],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.wave_prefix_depth_lift_over_uniform,
                expected_prefix_depth_lift_over_uniform[:, 3],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.wave_prefix_depth_to_uniform_ratio,
                expected_prefix_depth_to_uniform_ratio[:, 3],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(diagnostics.wave_defect_norm_sq, defect_norm_sq[:, 3], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.wave_defect_norm_sq_normalized,
                expected_defect_norm_sq_normalized[:, 3],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.wave_coherence_gate,
                edge_amplitude[:, 3],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.wave_coherence_gate_saturation_95,
                expected_saturation_95[:, 3],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.wave_coherence_gate_saturation_99,
                expected_saturation_99[:, 3],
                atol=1e-6,
            )
        )

    def test_collect_mnist_diagnostics_shapes_and_uncertainty(self) -> None:
        torch.manual_seed(7)
        model = MNISTReferenceModel(
            cell_config=CellConfig(
                hidden_dim=4,
                branch_depth=2,
                branch_base=2,
                num_layers=1,
                num_logic_heads=5,
            ),
            encoder_embedding_dim=24,
            encoder_base_channels=8,
        )
        outputs = model(torch.randn(2, 1, 28, 28, dtype=torch.float32))
        diagnostics = collect_mnist_diagnostics(outputs)

        self.assertEqual(diagnostics.core.edge_prefix_depth.shape, (2, 10))
        self.assertEqual(diagnostics.core.edge_defect_abs_mean.shape, (2, 10))
        self.assertEqual(diagnostics.core.edge_amplitude.shape, (2, 10))
        self.assertEqual(diagnostics.core.edge_prefix_depth_normalized.shape, (2, 10))
        self.assertEqual(diagnostics.core.edge_prefix_depth_uniform_baseline.shape, (2, 10))
        self.assertEqual(diagnostics.core.edge_defect_norm_sq_normalized.shape, (2, 10))
        self.assertEqual(diagnostics.core.edge_coherence_gate_saturation_95.shape, (2, 10))
        self.assertEqual(diagnostics.core.code_entropy.shape[:2], (2, 5))
        self.assertEqual(diagnostics.core.code_symbol_usage.shape, (2, 2))
        self.assertEqual(diagnostics.core.code_symbol_balance_per_depth.shape, (2,))
        self.assertEqual(diagnostics.class_confidence.shape, (2,))
        self.assertEqual(diagnostics.one_minus_confidence.shape, (2,))
        self.assertEqual(diagnostics.predicted_class.shape, (2,))
        self.assertTrue(
            torch.allclose(
                diagnostics.one_minus_confidence,
                1.0 - outputs.decode.class_probabilities.max(dim=-1).values,
                atol=1e-6,
            )
        )

    def test_serialization_and_logging_are_machine_readable(self) -> None:
        torch.manual_seed(7)
        model = MNISTReferenceModel(
            cell_config=CellConfig(
                hidden_dim=4,
                branch_depth=2,
                branch_base=2,
                num_layers=1,
                num_logic_heads=5,
            ),
            encoder_embedding_dim=24,
            encoder_base_channels=8,
        )
        diagnostics = collect_mnist_diagnostics(
            model(torch.randn(2, 1, 28, 28, dtype=torch.float32))
        )

        serializable = diagnostics.to_serializable()
        logging_summary = diagnostics.as_logging_dict()

        self.assertIsInstance(serializable, dict)
        self.assertIsInstance(serializable["core"], dict)
        self.assertIsInstance(serializable["class_confidence"], list)
        self.assertIsInstance(serializable["predicted_class"], list)
        self.assertIn("edge_prefix_depth_normalized", serializable["core"])
        self.assertIn("edge_prefix_depth_uniform_baseline", serializable["core"])
        self.assertIn("edge_defect_norm_sq_normalized", serializable["core"])
        self.assertIn("edge_coherence_gate_saturation_95", serializable["core"])
        self.assertIn("code_symbol_usage", serializable["core"])
        self.assertIn("code_symbol_balance_per_depth", serializable["core"])
        self.assertIn("wave_coherence_gate_mean", logging_summary)
        self.assertIn("one_minus_confidence_mean", logging_summary)
        self.assertIn("edge_prefix_depth_normalized_mean", logging_summary)
        self.assertIn("edge_prefix_depth_lift_over_uniform_mean", logging_summary)
        self.assertIn("edge_defect_norm_sq_normalized_mean", logging_summary)
        self.assertIn("edge_coherence_gate_saturation_rate_95", logging_summary)
        self.assertIn("code_symbol_balance_mean", logging_summary)
        self.assertIn("code_symbol_max_usage_max", logging_summary)
        self.assertIn("wave_prefix_depth_normalized_mean", logging_summary)
        self.assertIn("wave_defect_norm_sq_normalized_mean", logging_summary)
        self.assertIn("wave_coherence_gate_saturation_99_mean", logging_summary)
        self.assertTrue(all(isinstance(value, float) for value in logging_summary.values()))
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in logging_summary.values())
        )


if __name__ == "__main__":
    unittest.main()
