"""Diagnostics and logging hooks for reference pASCNN runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pascnn.models.mnist import MNISTModelOutputs
from pascnn.types import CellForwardOutputs
from pascnn.types import WAVE_EDGE_INDEX


def _tensor_to_serializable(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().tolist()


def _uniform_prefix_depth_baseline(
    digit_probabilities: torch.Tensor,
    edge_prefix_depth: torch.Tensor,
) -> torch.Tensor:
    branch_depth = digit_probabilities.shape[-2]
    branch_base = digit_probabilities.shape[-1]
    compute_dtype = edge_prefix_depth.dtype
    base = torch.tensor(
        float(branch_base),
        device=edge_prefix_depth.device,
        dtype=compute_dtype,
    )
    depth_index = torch.arange(
        1,
        branch_depth + 1,
        device=edge_prefix_depth.device,
        dtype=compute_dtype,
    )
    baseline_value = torch.pow(base, -depth_index).sum()
    return torch.full_like(edge_prefix_depth, fill_value=baseline_value)


def _saturation_mask(values: torch.Tensor, threshold: float) -> torch.Tensor:
    return (values >= threshold).to(dtype=values.dtype)


def _normalized_prefix_depth_from_outputs(
    cell_outputs: CellForwardOutputs,
) -> torch.Tensor:
    normalized_prefix_depth = cell_outputs.diagnostics.get("normalized_prefix_depth")
    if normalized_prefix_depth is not None:
        return normalized_prefix_depth.to(dtype=cell_outputs.prefix.prefix_depth.dtype)
    return cell_outputs.prefix.prefix_depth / float(cell_outputs.digit_probabilities.shape[-2])


def _normalized_defect_norm_sq_from_outputs(
    cell_outputs: CellForwardOutputs,
) -> torch.Tensor:
    normalized_defect_norm_sq = cell_outputs.diagnostics.get("normalized_defect_norm_sq")
    if normalized_defect_norm_sq is not None:
        return normalized_defect_norm_sq.to(
            dtype=cell_outputs.restriction.defect_norm_sq.dtype
        )
    return cell_outputs.restriction.defect_norm_sq / float(
        cell_outputs.vertex_state.shape[-1]
    )


@dataclass(frozen=True)
class CoreDiagnosticsSnapshot:
    edge_prefix_depth: torch.Tensor
    edge_prefix_depth_normalized: torch.Tensor
    edge_prefix_depth_uniform_baseline: torch.Tensor
    edge_prefix_depth_lift_over_uniform: torch.Tensor
    edge_prefix_depth_to_uniform_ratio: torch.Tensor
    edge_defect_abs_mean: torch.Tensor
    edge_defect_norm_sq: torch.Tensor
    edge_defect_norm_sq_normalized: torch.Tensor
    edge_amplitude: torch.Tensor
    edge_coherence_gate_saturation_95: torch.Tensor
    edge_coherence_gate_saturation_99: torch.Tensor
    edge_phase: torch.Tensor
    code_entropy: torch.Tensor
    code_entropy_mean: torch.Tensor
    code_symbol_usage: torch.Tensor
    code_symbol_balance_sq: torch.Tensor
    code_symbol_balance_per_depth: torch.Tensor
    code_symbol_max_usage_per_depth: torch.Tensor
    logic_probability_entropy: torch.Tensor
    wave_prefix_depth: torch.Tensor
    wave_prefix_depth_normalized: torch.Tensor
    wave_prefix_depth_uniform_baseline: torch.Tensor
    wave_prefix_depth_lift_over_uniform: torch.Tensor
    wave_prefix_depth_to_uniform_ratio: torch.Tensor
    wave_defect_abs_mean: torch.Tensor
    wave_defect_norm_sq: torch.Tensor
    wave_defect_norm_sq_normalized: torch.Tensor
    wave_coherence_gate: torch.Tensor
    wave_coherence_gate_saturation_95: torch.Tensor
    wave_coherence_gate_saturation_99: torch.Tensor

    def to_serializable(self) -> dict[str, Any]:
        return {
            "edge_prefix_depth": _tensor_to_serializable(self.edge_prefix_depth),
            "edge_prefix_depth_normalized": _tensor_to_serializable(
                self.edge_prefix_depth_normalized
            ),
            "edge_prefix_depth_uniform_baseline": _tensor_to_serializable(
                self.edge_prefix_depth_uniform_baseline
            ),
            "edge_prefix_depth_lift_over_uniform": _tensor_to_serializable(
                self.edge_prefix_depth_lift_over_uniform
            ),
            "edge_prefix_depth_to_uniform_ratio": _tensor_to_serializable(
                self.edge_prefix_depth_to_uniform_ratio
            ),
            "edge_defect_abs_mean": _tensor_to_serializable(self.edge_defect_abs_mean),
            "edge_defect_norm_sq": _tensor_to_serializable(self.edge_defect_norm_sq),
            "edge_defect_norm_sq_normalized": _tensor_to_serializable(
                self.edge_defect_norm_sq_normalized
            ),
            "edge_amplitude": _tensor_to_serializable(self.edge_amplitude),
            "edge_coherence_gate_saturation_95": _tensor_to_serializable(
                self.edge_coherence_gate_saturation_95
            ),
            "edge_coherence_gate_saturation_99": _tensor_to_serializable(
                self.edge_coherence_gate_saturation_99
            ),
            "edge_phase": _tensor_to_serializable(self.edge_phase),
            "code_entropy": _tensor_to_serializable(self.code_entropy),
            "code_entropy_mean": _tensor_to_serializable(self.code_entropy_mean),
            "code_symbol_usage": _tensor_to_serializable(self.code_symbol_usage),
            "code_symbol_balance_sq": _tensor_to_serializable(
                self.code_symbol_balance_sq
            ),
            "code_symbol_balance_per_depth": _tensor_to_serializable(
                self.code_symbol_balance_per_depth
            ),
            "code_symbol_max_usage_per_depth": _tensor_to_serializable(
                self.code_symbol_max_usage_per_depth
            ),
            "logic_probability_entropy": _tensor_to_serializable(
                self.logic_probability_entropy
            ),
            "wave_prefix_depth": _tensor_to_serializable(self.wave_prefix_depth),
            "wave_prefix_depth_normalized": _tensor_to_serializable(
                self.wave_prefix_depth_normalized
            ),
            "wave_prefix_depth_uniform_baseline": _tensor_to_serializable(
                self.wave_prefix_depth_uniform_baseline
            ),
            "wave_prefix_depth_lift_over_uniform": _tensor_to_serializable(
                self.wave_prefix_depth_lift_over_uniform
            ),
            "wave_prefix_depth_to_uniform_ratio": _tensor_to_serializable(
                self.wave_prefix_depth_to_uniform_ratio
            ),
            "wave_defect_abs_mean": _tensor_to_serializable(self.wave_defect_abs_mean),
            "wave_defect_norm_sq": _tensor_to_serializable(self.wave_defect_norm_sq),
            "wave_defect_norm_sq_normalized": _tensor_to_serializable(
                self.wave_defect_norm_sq_normalized
            ),
            "wave_coherence_gate": _tensor_to_serializable(self.wave_coherence_gate),
            "wave_coherence_gate_saturation_95": _tensor_to_serializable(
                self.wave_coherence_gate_saturation_95
            ),
            "wave_coherence_gate_saturation_99": _tensor_to_serializable(
                self.wave_coherence_gate_saturation_99
            ),
        }

    def as_logging_dict(self) -> dict[str, float]:
        return {
            "edge_prefix_depth_mean": float(self.edge_prefix_depth.mean().detach().cpu()),
            "edge_prefix_depth_normalized_mean": float(
                self.edge_prefix_depth_normalized.mean().detach().cpu()
            ),
            "edge_prefix_depth_uniform_baseline_mean": float(
                self.edge_prefix_depth_uniform_baseline.mean().detach().cpu()
            ),
            "edge_prefix_depth_lift_over_uniform_mean": float(
                self.edge_prefix_depth_lift_over_uniform.mean().detach().cpu()
            ),
            "edge_prefix_depth_to_uniform_ratio_mean": float(
                self.edge_prefix_depth_to_uniform_ratio.mean().detach().cpu()
            ),
            "edge_defect_abs_mean": float(self.edge_defect_abs_mean.mean().detach().cpu()),
            "edge_defect_norm_sq_mean": float(self.edge_defect_norm_sq.mean().detach().cpu()),
            "edge_defect_norm_sq_normalized_mean": float(
                self.edge_defect_norm_sq_normalized.mean().detach().cpu()
            ),
            "edge_amplitude_mean": float(self.edge_amplitude.mean().detach().cpu()),
            "edge_coherence_gate_saturation_rate_95": float(
                self.edge_coherence_gate_saturation_95.mean().detach().cpu()
            ),
            "edge_coherence_gate_saturation_rate_99": float(
                self.edge_coherence_gate_saturation_99.mean().detach().cpu()
            ),
            "edge_phase_abs_mean": float(self.edge_phase.abs().mean().detach().cpu()),
            "code_entropy_mean": float(self.code_entropy_mean.mean().detach().cpu()),
            "code_symbol_balance_mean": float(
                self.code_symbol_balance_per_depth.mean().detach().cpu()
            ),
            "code_symbol_max_usage_mean": float(
                self.code_symbol_max_usage_per_depth.mean().detach().cpu()
            ),
            "code_symbol_max_usage_max": float(
                self.code_symbol_max_usage_per_depth.max().detach().cpu()
            ),
            "logic_entropy_mean": float(
                self.logic_probability_entropy.mean().detach().cpu()
            ),
            "wave_prefix_depth_mean": float(self.wave_prefix_depth.mean().detach().cpu()),
            "wave_prefix_depth_normalized_mean": float(
                self.wave_prefix_depth_normalized.mean().detach().cpu()
            ),
            "wave_prefix_depth_uniform_baseline_mean": float(
                self.wave_prefix_depth_uniform_baseline.mean().detach().cpu()
            ),
            "wave_prefix_depth_lift_over_uniform_mean": float(
                self.wave_prefix_depth_lift_over_uniform.mean().detach().cpu()
            ),
            "wave_prefix_depth_to_uniform_ratio_mean": float(
                self.wave_prefix_depth_to_uniform_ratio.mean().detach().cpu()
            ),
            "wave_defect_abs_mean": float(
                self.wave_defect_abs_mean.mean().detach().cpu()
            ),
            "wave_defect_norm_sq_mean": float(
                self.wave_defect_norm_sq.mean().detach().cpu()
            ),
            "wave_defect_norm_sq_normalized_mean": float(
                self.wave_defect_norm_sq_normalized.mean().detach().cpu()
            ),
            "wave_coherence_gate_mean": float(
                self.wave_coherence_gate.mean().detach().cpu()
            ),
            "wave_coherence_gate_saturation_95_mean": float(
                self.wave_coherence_gate_saturation_95.mean().detach().cpu()
            ),
            "wave_coherence_gate_saturation_99_mean": float(
                self.wave_coherence_gate_saturation_99.mean().detach().cpu()
            ),
        }


@dataclass(frozen=True)
class MNISTDiagnosticsSnapshot:
    core: CoreDiagnosticsSnapshot
    class_confidence: torch.Tensor
    one_minus_confidence: torch.Tensor
    predicted_class: torch.Tensor

    def to_serializable(self) -> dict[str, Any]:
        return {
            "core": self.core.to_serializable(),
            "class_confidence": _tensor_to_serializable(self.class_confidence),
            "one_minus_confidence": _tensor_to_serializable(self.one_minus_confidence),
            "predicted_class": _tensor_to_serializable(self.predicted_class),
        }

    def as_logging_dict(self) -> dict[str, float]:
        summary = self.core.as_logging_dict()
        summary.update(
            {
                "class_confidence_mean": float(
                    self.class_confidence.mean().detach().cpu()
                ),
                "one_minus_confidence_mean": float(
                    self.one_minus_confidence.mean().detach().cpu()
                ),
            }
        )
        return summary


def collect_core_diagnostics(cell_outputs: CellForwardOutputs) -> CoreDiagnosticsSnapshot:
    if not isinstance(cell_outputs, CellForwardOutputs):
        raise TypeError("cell_outputs must be a CellForwardOutputs instance.")

    probability_epsilon = torch.finfo(cell_outputs.digit_probabilities.dtype).tiny
    digit_probabilities = cell_outputs.digit_probabilities.clamp_min(probability_epsilon)
    logic_probabilities = cell_outputs.readout.logic_probabilities.clamp_min(
        torch.finfo(cell_outputs.readout.logic_probabilities.dtype).tiny
    )

    edge_defect_abs_mean = cell_outputs.restriction.defect.abs().mean(dim=-1)
    code_entropy = -(digit_probabilities * digit_probabilities.log()).sum(dim=-1)
    code_entropy_mean = code_entropy.mean(dim=(-1, -2))
    branch_base = float(digit_probabilities.shape[-1])
    code_symbol_usage = digit_probabilities.mean(dim=(0, 1))
    code_symbol_balance_sq = (
        code_symbol_usage
        - torch.full_like(code_symbol_usage, fill_value=1.0 / branch_base)
    ).square()
    code_symbol_balance_per_depth = code_symbol_balance_sq.mean(dim=-1)
    code_symbol_max_usage_per_depth = code_symbol_usage.max(dim=-1).values
    logic_probability_entropy = -(logic_probabilities * logic_probabilities.log()).sum(
        dim=-1
    )
    edge_prefix_depth_normalized = _normalized_prefix_depth_from_outputs(cell_outputs)
    edge_defect_norm_sq_normalized = _normalized_defect_norm_sq_from_outputs(
        cell_outputs
    )
    edge_prefix_depth_uniform_baseline = _uniform_prefix_depth_baseline(
        digit_probabilities=digit_probabilities,
        edge_prefix_depth=cell_outputs.prefix.prefix_depth,
    )
    edge_prefix_depth_lift_over_uniform = (
        cell_outputs.prefix.prefix_depth - edge_prefix_depth_uniform_baseline
    )
    edge_prefix_depth_to_uniform_ratio = cell_outputs.prefix.prefix_depth / (
        edge_prefix_depth_uniform_baseline.clamp_min(probability_epsilon)
    )
    edge_coherence_gate_saturation_95 = _saturation_mask(
        cell_outputs.coherence.edge_amplitude,
        threshold=0.95,
    )
    edge_coherence_gate_saturation_99 = _saturation_mask(
        cell_outputs.coherence.edge_amplitude,
        threshold=0.99,
    )

    return CoreDiagnosticsSnapshot(
        edge_prefix_depth=cell_outputs.prefix.prefix_depth,
        edge_prefix_depth_normalized=edge_prefix_depth_normalized,
        edge_prefix_depth_uniform_baseline=edge_prefix_depth_uniform_baseline,
        edge_prefix_depth_lift_over_uniform=edge_prefix_depth_lift_over_uniform,
        edge_prefix_depth_to_uniform_ratio=edge_prefix_depth_to_uniform_ratio,
        edge_defect_abs_mean=edge_defect_abs_mean,
        edge_defect_norm_sq=cell_outputs.restriction.defect_norm_sq,
        edge_defect_norm_sq_normalized=edge_defect_norm_sq_normalized,
        edge_amplitude=cell_outputs.coherence.edge_amplitude,
        edge_coherence_gate_saturation_95=edge_coherence_gate_saturation_95,
        edge_coherence_gate_saturation_99=edge_coherence_gate_saturation_99,
        edge_phase=cell_outputs.coherence.edge_phase,
        code_entropy=code_entropy,
        code_entropy_mean=code_entropy_mean,
        code_symbol_usage=code_symbol_usage,
        code_symbol_balance_sq=code_symbol_balance_sq,
        code_symbol_balance_per_depth=code_symbol_balance_per_depth,
        code_symbol_max_usage_per_depth=code_symbol_max_usage_per_depth,
        logic_probability_entropy=logic_probability_entropy,
        wave_prefix_depth=cell_outputs.prefix.prefix_depth[:, WAVE_EDGE_INDEX],
        wave_prefix_depth_normalized=edge_prefix_depth_normalized[:, WAVE_EDGE_INDEX],
        wave_prefix_depth_uniform_baseline=edge_prefix_depth_uniform_baseline[
            :,
            WAVE_EDGE_INDEX,
        ],
        wave_prefix_depth_lift_over_uniform=edge_prefix_depth_lift_over_uniform[
            :,
            WAVE_EDGE_INDEX,
        ],
        wave_prefix_depth_to_uniform_ratio=edge_prefix_depth_to_uniform_ratio[
            :,
            WAVE_EDGE_INDEX,
        ],
        wave_defect_abs_mean=edge_defect_abs_mean[:, WAVE_EDGE_INDEX],
        wave_defect_norm_sq=cell_outputs.restriction.defect_norm_sq[:, WAVE_EDGE_INDEX],
        wave_defect_norm_sq_normalized=edge_defect_norm_sq_normalized[
            :,
            WAVE_EDGE_INDEX,
        ],
        wave_coherence_gate=cell_outputs.coherence.edge_amplitude[:, WAVE_EDGE_INDEX],
        wave_coherence_gate_saturation_95=edge_coherence_gate_saturation_95[
            :,
            WAVE_EDGE_INDEX,
        ],
        wave_coherence_gate_saturation_99=edge_coherence_gate_saturation_99[
            :,
            WAVE_EDGE_INDEX,
        ],
    )


def collect_mnist_diagnostics(
    model_outputs: MNISTModelOutputs,
) -> MNISTDiagnosticsSnapshot:
    if not isinstance(model_outputs, MNISTModelOutputs):
        raise TypeError("model_outputs must be an MNISTModelOutputs instance.")

    class_probabilities = model_outputs.decode.class_probabilities.clamp_min(
        torch.finfo(model_outputs.decode.class_probabilities.dtype).tiny
    )
    class_confidence, predicted_class = class_probabilities.max(dim=-1)
    one_minus_confidence = 1.0 - class_confidence

    return MNISTDiagnosticsSnapshot(
        core=collect_core_diagnostics(model_outputs.cell),
        class_confidence=class_confidence,
        one_minus_confidence=one_minus_confidence,
        predicted_class=predicted_class,
    )
