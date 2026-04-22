"""Generic one-cell pASCNN core skeleton."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd.profiler import record_function

from pascnn.core.readout import TernaryBornReadout
from pascnn.ops import complex_restrict_defect
from pascnn.ops import complex_scatter_add
from pascnn.ops import edge_coherence_uwca
from pascnn.ops import soft_prefix_match
from pascnn.types import CellConfig
from pascnn.types import CellForwardOutputs
from pascnn.types import CANONICAL_EDGE_TYPE_INDEX
from pascnn.types import EDGE_ENDPOINTS
from pascnn.types import EdgeCoherenceParameters
from pascnn.types import TransportMaps


class PASCNNCell(nn.Module):
    """Generic one-cell pASCNN core contract."""

    def __init__(self, config: CellConfig) -> None:
        super().__init__()
        self.config = config
        self.readout = TernaryBornReadout(config)
        self.state_real_init = nn.LazyLinear(
            config.num_vertices * config.hidden_dim,
        )
        self.state_imag_init = nn.LazyLinear(
            config.num_vertices * config.hidden_dim,
        )
        self.branch_logits_init = nn.LazyLinear(
            config.num_vertices * config.branch_depth * config.branch_base,
        )
        if config.num_layers > 1:
            with torch.random.fork_rng(devices=[]):
                self.branch_logits_state = nn.Linear(
                    2 * config.hidden_dim,
                    config.branch_depth * config.branch_base,
                )
            nn.init.zeros_(self.branch_logits_state.weight)
            nn.init.zeros_(self.branch_logits_state.bias)
        else:
            self.branch_logits_state = None

        self.source_transport_diagonal = nn.Parameter(
            torch.ones(
                3,
                config.hidden_dim,
                dtype=config.state_dtype,
            )
        )
        self.target_transport_diagonal = nn.Parameter(
            torch.ones(
                3,
                config.hidden_dim,
                dtype=config.state_dtype,
            )
        )
        self.self_diagonal = nn.Parameter(
            torch.ones(
                config.num_vertices,
                config.hidden_dim,
                dtype=config.state_dtype,
            )
        )
        self.phase_bias = nn.Parameter(torch.zeros(3, dtype=config.real_dtype))
        self.phase_scale = nn.Parameter(torch.zeros(3, dtype=config.real_dtype))
        self.amplitude_alpha = nn.Parameter(torch.ones(3, dtype=config.real_dtype))
        self.amplitude_threshold = nn.Parameter(
            torch.zeros(3, dtype=config.real_dtype)
        )
        self.defect_beta = nn.Parameter(
            torch.full(
                (3,),
                1.0 / float(config.hidden_dim),
                dtype=config.real_dtype,
            )
        )

        source_vertex_index = [edge[0] for edge in EDGE_ENDPOINTS]
        target_vertex_index = [edge[1] for edge in EDGE_ENDPOINTS]
        self.register_buffer(
            "edge_type_index",
            torch.tensor(CANONICAL_EDGE_TYPE_INDEX, dtype=torch.long),
        )
        self.register_buffer(
            "incidence_index",
            torch.tensor(EDGE_ENDPOINTS, dtype=torch.long),
        )
        self.register_buffer(
            "source_vertex_index",
            torch.tensor(source_vertex_index, dtype=torch.long),
        )
        self.register_buffer(
            "target_vertex_index",
            torch.tensor(target_vertex_index, dtype=torch.long),
        )

    def _init_vertex_state(self, h: torch.Tensor) -> torch.Tensor:
        batch_size = h.shape[0]
        real = self.state_real_init(h).reshape(
            batch_size,
            self.config.num_vertices,
            self.config.hidden_dim,
        )
        imag = self.state_imag_init(h).reshape(
            batch_size,
            self.config.num_vertices,
            self.config.hidden_dim,
        )
        vertex_state = torch.complex(
            real.to(dtype=self.config.real_dtype),
            imag.to(dtype=self.config.real_dtype),
        ).to(dtype=self.config.state_dtype)
        return vertex_state * (float(self.config.hidden_dim) ** -0.5)

    def _init_digit_logits(self, h: torch.Tensor) -> torch.Tensor:
        batch_size = h.shape[0]
        return self.branch_logits_init(h).reshape(
            batch_size,
            self.config.num_vertices,
            self.config.branch_depth,
            self.config.branch_base,
        ).to(dtype=self.config.real_dtype)

    def _digit_logits_from_vertex_state(self, vertex_state: torch.Tensor) -> torch.Tensor:
        if self.branch_logits_state is None:
            raise RuntimeError(
                "branch_logits_state is only defined when num_layers > 1."
            )
        state_features = torch.cat(
            (
                vertex_state.real.to(dtype=self.config.real_dtype),
                vertex_state.imag.to(dtype=self.config.real_dtype),
            ),
            dim=-1,
        )
        return self.branch_logits_state(state_features).reshape(
            vertex_state.shape[0],
            self.config.num_vertices,
            self.config.branch_depth,
            self.config.branch_base,
        )

    def _gather_edge_tensor(self, vertex_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_tensor = vertex_tensor.index_select(1, self.source_vertex_index)
        target_tensor = vertex_tensor.index_select(1, self.target_vertex_index)
        return source_tensor, target_tensor

    def _normalized_prefix_depth(self, prefix_depth: torch.Tensor) -> torch.Tensor:
        return prefix_depth / float(self.config.branch_depth)

    def _normalized_defect_norm_sq(self, defect_norm_sq: torch.Tensor) -> torch.Tensor:
        return defect_norm_sq / float(self.config.hidden_dim)

    def _transport_maps(self) -> TransportMaps:
        return TransportMaps(
            source_diagonal=self.source_transport_diagonal,
            target_diagonal=self.target_transport_diagonal,
        )

    def _coherence_parameters(self) -> EdgeCoherenceParameters:
        return EdgeCoherenceParameters(
            phase_bias=self.phase_bias,
            phase_scale=self.phase_scale,
            amplitude_alpha=self.amplitude_alpha,
            amplitude_threshold=self.amplitude_threshold,
            defect_beta=self.defect_beta,
        )

    def forward(self, h: torch.Tensor) -> CellForwardOutputs:
        if h.ndim != 2:
            raise ValueError("h must have shape [B, M].")
        if not h.is_floating_point():
            raise TypeError("h must be a floating tensor.")
        op_backend = self.config.op_backend

        with record_function("pascnn.init_branch_logit_prior"):
            digit_logit_prior = self._init_digit_logits(h)
        with record_function("pascnn.init_vertex_state"):
            vertex_state = self._init_vertex_state(h)

        prefix = None
        restriction = None
        coherence = None
        digit_logits = None
        digit_probabilities = None
        for _ in range(self.config.num_layers):
            with record_function("pascnn.dynamic_digit_logits"):
                if self.branch_logits_state is not None:
                    digit_logits = digit_logit_prior + self._digit_logits_from_vertex_state(
                        vertex_state
                    )
                else:
                    digit_logits = digit_logit_prior
                digit_probabilities = torch.softmax(digit_logits, dim=-1)
            with record_function("pascnn.gather_edge_digits"):
                source_digit_probs, target_digit_probs = self._gather_edge_tensor(
                    digit_probabilities
                )
            with record_function("pascnn.soft_prefix_match"):
                prefix = soft_prefix_match(
                    source_digit_probs,
                    target_digit_probs,
                    backend=op_backend,
                )

            with record_function("pascnn.gather_edge_state"):
                source_state, target_state = self._gather_edge_tensor(vertex_state)
            with record_function("pascnn.complex_restrict_defect"):
                restriction = complex_restrict_defect(
                    source_state=source_state,
                    target_state=target_state,
                    edge_type_index=self.edge_type_index,
                    transport_maps=self._transport_maps(),
                    backend=op_backend,
                )
            with record_function("pascnn.normalize_prefix_and_defect"):
                normalized_prefix_depth = self._normalized_prefix_depth(prefix.prefix_depth)
                normalized_defect_norm_sq = self._normalized_defect_norm_sq(
                    restriction.defect_norm_sq
                )
            with record_function("pascnn.edge_coherence_uwca"):
                coherence = edge_coherence_uwca(
                    source_transport=restriction.source_transport,
                    target_transport=restriction.target_transport,
                    prefix_depth=prefix.prefix_depth,
                    defect_norm_sq=restriction.defect_norm_sq,
                    edge_type_index=self.edge_type_index,
                    coherence=self._coherence_parameters(),
                    backend=op_backend,
                )
            with record_function("pascnn.complex_scatter_add"):
                vertex_message_sum = complex_scatter_add(
                    edge_messages=coherence.edge_message,
                    incidence_index=self.incidence_index,
                    num_vertices=self.config.num_vertices,
                    backend=op_backend,
                )
            with record_function("pascnn.vertex_update"):
                self_update = self.self_diagonal.unsqueeze(0) * vertex_state
                vertex_state = self_update + vertex_message_sum

        assert prefix is not None
        assert restriction is not None
        assert coherence is not None
        assert digit_logits is not None
        assert digit_probabilities is not None
        with record_function("pascnn.ternary_born_readout"):
            readout = self.readout(vertex_state)

        diagnostics = {
            "edge_type_index": self.edge_type_index,
            "incidence_index": self.incidence_index,
            "prefix_depth": prefix.prefix_depth,
            "normalized_prefix_depth": normalized_prefix_depth,
            "defect_norm_sq": restriction.defect_norm_sq,
            "normalized_defect_norm_sq": normalized_defect_norm_sq,
            "edge_amplitude": coherence.edge_amplitude,
            "edge_phase": coherence.edge_phase,
            "edge_coefficient_abs": coherence.edge_coefficient.abs(),
            "logic_amplitude_abs": readout.logic_amplitudes.abs(),
            "logic_probability_sum": readout.logic_probabilities.sum(dim=-1),
        }

        return CellForwardOutputs(
            vertex_state=vertex_state,
            digit_logits=digit_logits,
            digit_probabilities=digit_probabilities,
            prefix=prefix,
            restriction=restriction,
            coherence=coherence,
            readout=readout,
            diagnostics=diagnostics,
        )
