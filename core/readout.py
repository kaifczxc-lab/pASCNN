"""Logical ternary readout skeleton."""

from __future__ import annotations

import torch
from torch import nn

from pascnn.types import CellConfig
from pascnn.types import LOGIC_VERTEX_INDICES
from pascnn.types import LogicalReadoutOutputs


class TernaryBornReadout(nn.Module):
    """Ternary Born-style readout contract."""

    def __init__(self, config: CellConfig) -> None:
        super().__init__()
        self.config = config
        self.logic_weight = nn.Parameter(
            torch.ones(
                config.num_logic_heads,
                len(LOGIC_VERTEX_INDICES),
                config.hidden_dim,
                dtype=config.state_dtype,
            )
        )
        self._probability_epsilon = torch.finfo(config.real_dtype).tiny

    def forward(self, vertex_state: torch.Tensor) -> LogicalReadoutOutputs:
        if vertex_state.ndim != 3:
            raise ValueError("vertex_state must have shape [B, V, D]")
        if vertex_state.shape[1] != self.config.num_vertices:
            raise ValueError("vertex_state must match the configured number of vertices")
        if vertex_state.shape[2] != self.config.hidden_dim:
            raise ValueError("vertex_state must match the configured hidden dim")
        if not vertex_state.is_complex():
            raise TypeError("vertex_state must be a complex tensor")

        logical_vertex_state = vertex_state[:, LOGIC_VERTEX_INDICES, :]
        logic_amplitudes = torch.einsum(
            "hcd,bcd->bhc",
            self.logic_weight.conj(),
            logical_vertex_state,
        )
        logic_energy = logic_amplitudes.abs().square()
        normalization = logic_energy.sum(dim=-1, keepdim=True).clamp_min(
            self._probability_epsilon
        )
        logic_probabilities = logic_energy / normalization

        return LogicalReadoutOutputs(
            logic_amplitudes=logic_amplitudes,
            logic_probabilities=logic_probabilities.to(dtype=self.config.real_dtype),
        )
