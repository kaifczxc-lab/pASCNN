"""Typed API contracts for the pASCNN runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


VertexLabel = Literal["L", "R", "-1", "0", "+1"]
EdgeType = Literal["logic", "wave", "cross"]
OpBackend = Literal["reference", "native", "auto"]

VERTEX_LABELS: tuple[VertexLabel, ...] = ("L", "R", "-1", "0", "+1")
EDGE_TYPE_LABELS: tuple[EdgeType, ...] = ("logic", "wave", "cross")
EDGE_TYPE_INDEX: dict[EdgeType, int] = {
    "logic": 0,
    "wave": 1,
    "cross": 2,
}
VERTEX_INDEX: dict[VertexLabel, int] = {
    "L": 0,
    "R": 1,
    "-1": 2,
    "0": 3,
    "+1": 4,
}
LOGIC_VERTEX_INDICES: tuple[int, int, int] = (
    VERTEX_INDEX["-1"],
    VERTEX_INDEX["0"],
    VERTEX_INDEX["+1"],
)
LOGIC_EDGE_INDICES: tuple[int, int, int] = (0, 1, 2)
WAVE_EDGE_INDEX: int = 3
CROSS_EDGE_INDICES: tuple[int, ...] = (4, 5, 6, 7, 8, 9)

EDGE_ENDPOINTS: tuple[tuple[int, int], ...] = (
    (VERTEX_INDEX["-1"], VERTEX_INDEX["0"]),
    (VERTEX_INDEX["0"], VERTEX_INDEX["+1"]),
    (VERTEX_INDEX["+1"], VERTEX_INDEX["-1"]),
    (VERTEX_INDEX["L"], VERTEX_INDEX["R"]),
    (VERTEX_INDEX["-1"], VERTEX_INDEX["L"]),
    (VERTEX_INDEX["-1"], VERTEX_INDEX["R"]),
    (VERTEX_INDEX["0"], VERTEX_INDEX["L"]),
    (VERTEX_INDEX["0"], VERTEX_INDEX["R"]),
    (VERTEX_INDEX["+1"], VERTEX_INDEX["L"]),
    (VERTEX_INDEX["+1"], VERTEX_INDEX["R"]),
)
EDGE_TYPES: tuple[EdgeType, ...] = (
    "logic",
    "logic",
    "logic",
    "wave",
    "cross",
    "cross",
    "cross",
    "cross",
    "cross",
    "cross",
)
CANONICAL_EDGE_TYPE_INDEX: tuple[int, ...] = tuple(
    EDGE_TYPE_INDEX[edge_type]
    for edge_type in EDGE_TYPES
)


@dataclass(frozen=True)
class CellConfig:
    hidden_dim: int
    branch_depth: int
    branch_base: int
    num_layers: int
    num_logic_heads: int = 1
    op_backend: OpBackend = "reference"
    edge_map_mode: Literal["typed_diagonal", "typed_low_rank"] = "typed_diagonal"
    state_dtype: torch.dtype = torch.complex64
    real_dtype: torch.dtype = torch.float32
    num_vertices: int = len(VERTEX_LABELS)
    num_edges: int = len(EDGE_ENDPOINTS)


@dataclass(frozen=True)
class TransportMaps:
    source_diagonal: torch.Tensor
    target_diagonal: torch.Tensor


@dataclass(frozen=True)
class EdgeCoherenceParameters:
    phase_bias: torch.Tensor
    phase_scale: torch.Tensor
    amplitude_alpha: torch.Tensor
    amplitude_threshold: torch.Tensor
    defect_beta: torch.Tensor


@dataclass(frozen=True)
class VertexUpdateParameters:
    self_diagonal: torch.Tensor
    scatter_diagonal: torch.Tensor | None = None


@dataclass(frozen=True)
class PrefixMatchOutputs:
    position_agreement: torch.Tensor
    prefix_compatibility: torch.Tensor
    prefix_depth: torch.Tensor


@dataclass(frozen=True)
class RestrictDefectOutputs:
    source_transport: torch.Tensor
    target_transport: torch.Tensor
    defect: torch.Tensor
    defect_norm_sq: torch.Tensor


@dataclass(frozen=True)
class EdgeCoherenceOutputs:
    edge_mean: torch.Tensor
    edge_amplitude: torch.Tensor
    edge_phase: torch.Tensor
    edge_coefficient: torch.Tensor
    edge_message: torch.Tensor


@dataclass(frozen=True)
class LogicalReadoutOutputs:
    logic_amplitudes: torch.Tensor
    logic_probabilities: torch.Tensor


@dataclass(frozen=True)
class CellForwardOutputs:
    vertex_state: torch.Tensor
    digit_logits: torch.Tensor
    digit_probabilities: torch.Tensor
    prefix: PrefixMatchOutputs
    restriction: RestrictDefectOutputs
    coherence: EdgeCoherenceOutputs
    readout: LogicalReadoutOutputs
    diagnostics: dict[str, torch.Tensor]
