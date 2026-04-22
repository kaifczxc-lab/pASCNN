import unittest

import torch

from pascnn.core.cell import PASCNNCell
from pascnn.types import CellConfig


class PASCNNCellTests(unittest.TestCase):
    def test_forward_shapes_and_diagnostics(self) -> None:
        torch.manual_seed(7)
        cell = PASCNNCell(
            CellConfig(
                hidden_dim=4,
                branch_depth=3,
                branch_base=2,
                num_layers=2,
                num_logic_heads=2,
            )
        )
        h = torch.randn(2, 6)

        outputs = cell(h)

        self.assertEqual(outputs.vertex_state.shape, (2, 5, 4))
        self.assertEqual(outputs.digit_logits.shape, (2, 5, 3, 2))
        self.assertEqual(outputs.digit_probabilities.shape, (2, 5, 3, 2))
        self.assertEqual(outputs.prefix.position_agreement.shape, (2, 10, 3))
        self.assertEqual(outputs.prefix.prefix_depth.shape, (2, 10))
        self.assertEqual(outputs.restriction.source_transport.shape, (2, 10, 4))
        self.assertEqual(outputs.restriction.defect_norm_sq.shape, (2, 10))
        self.assertEqual(outputs.coherence.edge_message.shape, (2, 10, 4))
        self.assertEqual(outputs.readout.logic_amplitudes.shape, (2, 2, 3))
        self.assertEqual(outputs.readout.logic_probabilities.shape, (2, 2, 3))
        self.assertIn("prefix_depth", outputs.diagnostics)
        self.assertIn("normalized_prefix_depth", outputs.diagnostics)
        self.assertIn("defect_norm_sq", outputs.diagnostics)
        self.assertIn("normalized_defect_norm_sq", outputs.diagnostics)
        self.assertIn("edge_amplitude", outputs.diagnostics)
        self.assertIn("edge_phase", outputs.diagnostics)
        self.assertIn("logic_amplitude_abs", outputs.diagnostics)
        self.assertIn("logic_probability_sum", outputs.diagnostics)
        self.assertEqual(outputs.diagnostics["edge_type_index"].shape, (10,))
        self.assertEqual(outputs.diagnostics["incidence_index"].shape, (10, 2))
        self.assertEqual(outputs.diagnostics["normalized_prefix_depth"].shape, (2, 10))
        self.assertEqual(outputs.diagnostics["normalized_defect_norm_sq"].shape, (2, 10))
        self.assertTrue(
            torch.allclose(
                outputs.readout.logic_probabilities.sum(dim=-1),
                torch.ones(2, 2),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.isfinite(outputs.vertex_state.real).all())
        self.assertTrue(torch.isfinite(outputs.vertex_state.imag).all())

    def test_backward_reaches_input_and_core_parameters(self) -> None:
        torch.manual_seed(7)
        cell = PASCNNCell(
            CellConfig(
                hidden_dim=3,
                branch_depth=4,
                branch_base=3,
                num_layers=2,
                num_logic_heads=1,
            )
        )
        h = torch.randn(2, 5, requires_grad=True)

        outputs = cell(h)
        loss = (
            outputs.vertex_state.abs().mean()
            + outputs.prefix.prefix_depth.mean()
            + outputs.restriction.defect_norm_sq.mean()
            + outputs.coherence.edge_amplitude.mean()
            + outputs.readout.logic_amplitudes.abs().mean()
            + outputs.readout.logic_probabilities[..., 0].mean()
        )
        loss.backward()

        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertIsNotNone(cell.state_real_init.weight.grad)
        self.assertIsNotNone(cell.branch_logits_init.weight.grad)
        self.assertIsNotNone(cell.branch_logits_state.weight.grad)
        self.assertIsNotNone(cell.source_transport_diagonal.grad)
        self.assertIsNotNone(cell.phase_bias.grad)
        self.assertIsNotNone(cell.readout.logic_weight.grad)
        self.assertTrue(torch.isfinite(cell.state_real_init.weight.grad).all())
        self.assertTrue(torch.isfinite(cell.branch_logits_init.weight.grad).all())
        self.assertTrue(torch.isfinite(cell.branch_logits_state.weight.grad).all())
        self.assertTrue(torch.isfinite(cell.source_transport_diagonal.grad).all())
        self.assertTrue(torch.isfinite(cell.phase_bias.grad).all())
        self.assertTrue(torch.isfinite(cell.readout.logic_weight.grad).all())

    def test_branch_logits_update_across_layers_from_vertex_state(self) -> None:
        torch.manual_seed(7)
        cell = PASCNNCell(
            CellConfig(
                hidden_dim=4,
                branch_depth=3,
                branch_base=2,
                num_layers=2,
                num_logic_heads=2,
            )
        )
        h = torch.randn(2, 6)
        with torch.no_grad():
            cell.branch_logits_state.weight.fill_(0.125)
            cell.branch_logits_state.bias.fill_(0.05)

        initial_vertex_state = cell._init_vertex_state(h)
        initial_digit_logit_prior = cell._init_digit_logits(h)
        first_layer_digit_logits = initial_digit_logit_prior + cell._digit_logits_from_vertex_state(
            initial_vertex_state
        )

        outputs = cell(h)

        self.assertFalse(
            torch.allclose(
                outputs.digit_logits,
                initial_digit_logit_prior,
                atol=1e-6,
            )
        )
        self.assertFalse(
            torch.allclose(
                outputs.digit_logits,
                first_layer_digit_logits,
                atol=1e-6,
            )
        )

    def test_rejects_non_matrix_input(self) -> None:
        cell = PASCNNCell(
            CellConfig(
                hidden_dim=2,
                branch_depth=2,
                branch_base=2,
                num_layers=1,
                num_logic_heads=1,
            )
        )
        h = torch.randn(2, 3, 4)

        with self.assertRaisesRegex(
            ValueError,
            "h must have shape \\[B, M\\]",
        ):
            cell(h)


if __name__ == "__main__":
    unittest.main()
