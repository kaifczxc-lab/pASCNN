"""Pure PyTorch reference-op skeletons for pASCNN."""

from __future__ import annotations

import torch

from pascnn.types import EdgeCoherenceOutputs
from pascnn.types import EdgeCoherenceParameters
from pascnn.types import PrefixMatchOutputs
from pascnn.types import RestrictDefectOutputs
from pascnn.types import TransportMaps


def _validate_soft_prefix_inputs(
    source_digit_probs: torch.Tensor,
    target_digit_probs: torch.Tensor,
) -> None:
    if source_digit_probs.ndim != 4:
        raise ValueError(
            "source_digit_probs must have shape [B, E, K, P]."
        )
    if target_digit_probs.ndim != 4:
        raise ValueError(
            "target_digit_probs must have shape [B, E, K, P]."
        )
    if source_digit_probs.shape != target_digit_probs.shape:
        raise ValueError(
            "source_digit_probs and target_digit_probs must have the same shape."
        )
    if not source_digit_probs.is_floating_point():
        raise TypeError("source_digit_probs must be a floating tensor.")
    if not target_digit_probs.is_floating_point():
        raise TypeError("target_digit_probs must be a floating tensor.")
    if source_digit_probs.shape[-2] < 1:
        raise ValueError("branch depth K must be at least 1.")
    if source_digit_probs.shape[-1] < 2:
        raise ValueError("branch base P must be at least 2.")


def _validate_complex_restrict_inputs(
    source_state: torch.Tensor,
    target_state: torch.Tensor,
    edge_type_index: torch.Tensor,
    transport_maps: TransportMaps,
) -> None:
    if source_state.ndim != 3:
        raise ValueError("source_state must have shape [B, E, D].")
    if target_state.ndim != 3:
        raise ValueError("target_state must have shape [B, E, D].")
    if source_state.shape != target_state.shape:
        raise ValueError("source_state and target_state must have the same shape.")
    if not source_state.is_complex():
        raise TypeError("source_state must be a complex tensor.")
    if not target_state.is_complex():
        raise TypeError("target_state must be a complex tensor.")

    if edge_type_index.ndim not in (1, 2):
        raise ValueError("edge_type_index must have shape [E] or [B, E].")
    if edge_type_index.ndim == 1 and edge_type_index.shape[0] != source_state.shape[1]:
        raise ValueError("edge_type_index with shape [E] must match the edge axis.")
    if edge_type_index.ndim == 2 and edge_type_index.shape != source_state.shape[:2]:
        raise ValueError("edge_type_index with shape [B, E] must match state axes.")
    if edge_type_index.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("edge_type_index must be an integer tensor.")

    if transport_maps.source_diagonal.ndim != 2:
        raise ValueError(
            "transport_maps.source_diagonal must have shape [T, D]."
        )
    if transport_maps.target_diagonal.ndim != 2:
        raise ValueError(
            "transport_maps.target_diagonal must have shape [T, D]."
        )
    if transport_maps.source_diagonal.shape != transport_maps.target_diagonal.shape:
        raise ValueError(
            "source_diagonal and target_diagonal must have the same shape."
        )
    if transport_maps.source_diagonal.shape[1] != source_state.shape[-1]:
        raise ValueError("transport map hidden dim D must match state hidden dim D.")
    if not transport_maps.source_diagonal.is_complex():
        raise TypeError("transport_maps.source_diagonal must be a complex tensor.")
    if not transport_maps.target_diagonal.is_complex():
        raise TypeError("transport_maps.target_diagonal must be a complex tensor.")

    min_index = int(edge_type_index.min().item())
    max_index = int(edge_type_index.max().item())
    if min_index < 0:
        raise ValueError("edge_type_index must be non-negative.")
    if max_index >= transport_maps.source_diagonal.shape[0]:
        raise ValueError("edge_type_index exceeds the number of typed transport maps.")


def _validate_edge_coherence_inputs(
    source_transport: torch.Tensor,
    target_transport: torch.Tensor,
    prefix_depth: torch.Tensor,
    defect_norm_sq: torch.Tensor,
    edge_type_index: torch.Tensor,
    coherence: EdgeCoherenceParameters,
) -> None:
    if source_transport.ndim != 3:
        raise ValueError("source_transport must have shape [B, E, D].")
    if target_transport.ndim != 3:
        raise ValueError("target_transport must have shape [B, E, D].")
    if source_transport.shape != target_transport.shape:
        raise ValueError(
            "source_transport and target_transport must have the same shape."
        )
    if not source_transport.is_complex():
        raise TypeError("source_transport must be a complex tensor.")
    if not target_transport.is_complex():
        raise TypeError("target_transport must be a complex tensor.")

    if prefix_depth.ndim != 2:
        raise ValueError("prefix_depth must have shape [B, E].")
    if defect_norm_sq.ndim != 2:
        raise ValueError("defect_norm_sq must have shape [B, E].")
    if prefix_depth.shape != defect_norm_sq.shape:
        raise ValueError("prefix_depth and defect_norm_sq must have the same shape.")
    if prefix_depth.shape != source_transport.shape[:2]:
        raise ValueError("prefix_depth and defect_norm_sq must match [B, E].")
    if not prefix_depth.is_floating_point():
        raise TypeError("prefix_depth must be a floating tensor.")
    if not defect_norm_sq.is_floating_point():
        raise TypeError("defect_norm_sq must be a floating tensor.")

    if edge_type_index.ndim not in (1, 2):
        raise ValueError("edge_type_index must have shape [E] or [B, E].")
    if edge_type_index.ndim == 1 and edge_type_index.shape[0] != source_transport.shape[1]:
        raise ValueError("edge_type_index with shape [E] must match the edge axis.")
    if edge_type_index.ndim == 2 and edge_type_index.shape != source_transport.shape[:2]:
        raise ValueError("edge_type_index with shape [B, E] must match transport axes.")
    if edge_type_index.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("edge_type_index must be an integer tensor.")

    coherence_fields = (
        coherence.phase_bias,
        coherence.phase_scale,
        coherence.amplitude_alpha,
        coherence.amplitude_threshold,
        coherence.defect_beta,
    )
    if any(field.ndim != 1 for field in coherence_fields):
        raise ValueError("all coherence parameter tensors must have shape [T].")
    reference_shape = coherence.phase_bias.shape
    if any(field.shape != reference_shape for field in coherence_fields):
        raise ValueError("all coherence parameter tensors must share shape [T].")
    if any((not field.is_floating_point()) or field.is_complex() for field in coherence_fields):
        raise TypeError("all coherence parameter tensors must be real floating tensors.")

    min_index = int(edge_type_index.min().item())
    max_index = int(edge_type_index.max().item())
    if min_index < 0:
        raise ValueError("edge_type_index must be non-negative.")
    if max_index >= coherence.phase_bias.shape[0]:
        raise ValueError(
            "edge_type_index exceeds the number of typed coherence parameter sets."
        )


def _validate_complex_scatter_inputs(
    edge_messages: torch.Tensor,
    incidence_index: torch.Tensor,
    scatter_gains: torch.Tensor | None,
    num_vertices: int,
) -> None:
    if edge_messages.ndim != 3:
        raise ValueError("edge_messages must have shape [B, E, D].")
    if not edge_messages.is_complex():
        raise TypeError("edge_messages must be a complex tensor.")
    if num_vertices < 1:
        raise ValueError("num_vertices must be at least 1.")

    if incidence_index.ndim not in (2, 3):
        raise ValueError("incidence_index must have shape [E, 2] or [B, E, 2].")
    if incidence_index.shape[-1] != 2:
        raise ValueError("incidence_index must have a final dimension of size 2.")
    if incidence_index.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("incidence_index must be an integer tensor.")

    if incidence_index.ndim == 2 and incidence_index.shape[0] != edge_messages.shape[1]:
        raise ValueError("incidence_index with shape [E, 2] must match the edge axis.")
    if incidence_index.ndim == 3 and incidence_index.shape[:2] != edge_messages.shape[:2]:
        raise ValueError(
            "incidence_index with shape [B, E, 2] must match message batch and edge axes."
        )

    min_index = int(incidence_index.min().item())
    max_index = int(incidence_index.max().item())
    if min_index < 0:
        raise ValueError("incidence_index must be non-negative.")
    if max_index >= num_vertices:
        raise ValueError("incidence_index exceeds num_vertices.")

    if scatter_gains is None:
        return

    if scatter_gains.ndim not in (2, 3, 4):
        raise ValueError(
            "scatter_gains must have shape [E, 2], [E, 2, D], or [B, E, 2, D]."
        )
    if scatter_gains.ndim == 2 and scatter_gains.shape[0] != edge_messages.shape[1]:
        raise ValueError("scatter_gains with shape [E, 2] must match the edge axis.")
    if scatter_gains.ndim == 2 and scatter_gains.shape[1] != 2:
        raise ValueError("scatter_gains with shape [E, 2] must have size 2 on axis 1.")
    if scatter_gains.ndim == 3:
        if scatter_gains.shape[1] != 2:
            raise ValueError("scatter_gains must have a penultimate dimension of size 2.")
        if scatter_gains.shape[0] != edge_messages.shape[1]:
            raise ValueError(
                "scatter_gains with shape [E, 2, D] must match the edge axis."
            )
        if scatter_gains.shape[-1] != edge_messages.shape[-1]:
            raise ValueError(
                "scatter_gains with shape [E, 2, D] must match message hidden dim D."
            )
    if scatter_gains.ndim == 4:
        if scatter_gains.shape[-2] != 2:
            raise ValueError("scatter_gains must have a penultimate dimension of size 2.")
        if scatter_gains.shape[:2] != edge_messages.shape[:2]:
            raise ValueError(
                "scatter_gains with shape [B, E, 2, D] must match message batch and edge axes."
            )
        if scatter_gains.shape[-1] != edge_messages.shape[-1]:
            raise ValueError(
                "scatter_gains with shape [B, E, 2, D] must match message hidden dim D."
            )


def soft_prefix_match(
    source_digit_probs: torch.Tensor,
    target_digit_probs: torch.Tensor,
) -> PrefixMatchOutputs:
    """Return the differentiable prefix-match contract."""

    _validate_soft_prefix_inputs(source_digit_probs, target_digit_probs)

    compute_dtype = torch.promote_types(
        source_digit_probs.dtype,
        target_digit_probs.dtype,
    )
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    source = source_digit_probs.to(dtype=compute_dtype)
    target = target_digit_probs.to(dtype=compute_dtype)

    position_agreement = (source * target).sum(dim=-1)
    prefix_compatibility = torch.cumprod(position_agreement, dim=-1)
    prefix_depth = prefix_compatibility.sum(dim=-1)

    return PrefixMatchOutputs(
        position_agreement=position_agreement,
        prefix_compatibility=prefix_compatibility,
        prefix_depth=prefix_depth,
    )


def complex_restrict_defect(
    source_state: torch.Tensor,
    target_state: torch.Tensor,
    edge_type_index: torch.Tensor,
    transport_maps: TransportMaps,
) -> RestrictDefectOutputs:
    """Return transported edge states and their defect."""

    _validate_complex_restrict_inputs(
        source_state,
        target_state,
        edge_type_index,
        transport_maps,
    )

    compute_dtype = torch.promote_types(source_state.dtype, target_state.dtype)
    compute_dtype = torch.promote_types(
        compute_dtype,
        transport_maps.source_diagonal.dtype,
    )
    compute_dtype = torch.promote_types(
        compute_dtype,
        transport_maps.target_diagonal.dtype,
    )

    source = source_state.to(dtype=compute_dtype)
    target = target_state.to(dtype=compute_dtype)
    source_diagonal = transport_maps.source_diagonal.to(
        device=source.device,
        dtype=compute_dtype,
    )
    target_diagonal = transport_maps.target_diagonal.to(
        device=target.device,
        dtype=compute_dtype,
    )
    edge_type_index = edge_type_index.to(device=source.device, dtype=torch.long)

    selected_source_diagonal = source_diagonal[edge_type_index]
    selected_target_diagonal = target_diagonal[edge_type_index]

    source_transport = selected_source_diagonal * source
    target_transport = selected_target_diagonal * target
    defect = source_transport - target_transport
    defect_norm_sq = defect.abs().square().sum(dim=-1)

    return RestrictDefectOutputs(
        source_transport=source_transport,
        target_transport=target_transport,
        defect=defect,
        defect_norm_sq=defect_norm_sq,
    )


def edge_coherence_uwca(
    source_transport: torch.Tensor,
    target_transport: torch.Tensor,
    prefix_depth: torch.Tensor,
    defect_norm_sq: torch.Tensor,
    edge_type_index: torch.Tensor,
    coherence: EdgeCoherenceParameters,
) -> EdgeCoherenceOutputs:
    """Return amplitude, phase, coefficient, and fused edge messages."""

    _validate_edge_coherence_inputs(
        source_transport,
        target_transport,
        prefix_depth,
        defect_norm_sq,
        edge_type_index,
        coherence,
    )

    complex_compute_dtype = torch.promote_types(
        source_transport.dtype,
        target_transport.dtype,
    )
    real_compute_dtype = torch.promote_types(prefix_depth.dtype, defect_norm_sq.dtype)
    for parameter in (
        coherence.phase_bias,
        coherence.phase_scale,
        coherence.amplitude_alpha,
        coherence.amplitude_threshold,
        coherence.defect_beta,
    ):
        real_compute_dtype = torch.promote_types(real_compute_dtype, parameter.dtype)
    if real_compute_dtype in (torch.float16, torch.bfloat16):
        real_compute_dtype = torch.float32

    source = source_transport.to(dtype=complex_compute_dtype)
    target = target_transport.to(dtype=complex_compute_dtype)
    prefix_depth = prefix_depth.to(device=source.device, dtype=real_compute_dtype)
    defect_norm_sq = defect_norm_sq.to(device=source.device, dtype=real_compute_dtype)
    edge_type_index = edge_type_index.to(device=source.device, dtype=torch.long)

    phase_bias = coherence.phase_bias.to(device=source.device, dtype=real_compute_dtype)
    phase_scale = coherence.phase_scale.to(
        device=source.device,
        dtype=real_compute_dtype,
    )
    amplitude_alpha = coherence.amplitude_alpha.to(
        device=source.device,
        dtype=real_compute_dtype,
    )
    amplitude_threshold = coherence.amplitude_threshold.to(
        device=source.device,
        dtype=real_compute_dtype,
    )
    defect_beta = coherence.defect_beta.to(
        device=source.device,
        dtype=real_compute_dtype,
    )

    selected_phase_bias = phase_bias[edge_type_index]
    selected_phase_scale = phase_scale[edge_type_index]
    selected_amplitude_alpha = amplitude_alpha[edge_type_index]
    selected_amplitude_threshold = amplitude_threshold[edge_type_index]
    selected_defect_beta = defect_beta[edge_type_index].abs()

    edge_mean = (source + target) * 0.5
    edge_phase = selected_phase_bias + selected_phase_scale * prefix_depth
    edge_amplitude = torch.sigmoid(
        selected_amplitude_alpha * (prefix_depth - selected_amplitude_threshold)
    ) * torch.exp(-selected_defect_beta * defect_norm_sq)
    edge_coefficient = torch.polar(edge_amplitude, edge_phase).to(
        dtype=complex_compute_dtype
    )
    edge_message = edge_mean * edge_coefficient.unsqueeze(-1)

    return EdgeCoherenceOutputs(
        edge_mean=edge_mean,
        edge_amplitude=edge_amplitude,
        edge_phase=edge_phase,
        edge_coefficient=edge_coefficient,
        edge_message=edge_message,
    )


def complex_scatter_add(
    edge_messages: torch.Tensor,
    incidence_index: torch.Tensor,
    scatter_gains: torch.Tensor | None = None,
    num_vertices: int = 5,
) -> torch.Tensor:
    """Scatter edge messages back to vertices."""

    _validate_complex_scatter_inputs(
        edge_messages,
        incidence_index,
        scatter_gains,
        num_vertices,
    )

    compute_dtype = edge_messages.dtype
    messages = edge_messages.to(dtype=compute_dtype)
    incidence_index = incidence_index.to(device=messages.device, dtype=torch.long)

    if incidence_index.ndim == 2:
        incidence_index = incidence_index.unsqueeze(0).expand(messages.shape[0], -1, -1)

    if scatter_gains is None:
        gains = None
    else:
        gains = scatter_gains.to(device=messages.device)
        if gains.ndim == 2:
            gains = gains.unsqueeze(0).unsqueeze(-1).expand(
                messages.shape[0],
                -1,
                -1,
                messages.shape[-1],
            )
        elif gains.ndim == 3:
            gains = gains.unsqueeze(0).expand(messages.shape[0], -1, -1, -1)

        if not gains.is_complex():
            gains = gains.to(dtype=messages.real.dtype)

    vertex_updates = torch.zeros(
        messages.shape[0],
        num_vertices,
        messages.shape[-1],
        dtype=compute_dtype,
        device=messages.device,
    )

    for batch_index in range(messages.shape[0]):
        for endpoint_slot in range(2):
            endpoint_messages = messages[batch_index]
            if gains is not None:
                endpoint_messages = endpoint_messages * gains[batch_index, :, endpoint_slot, :]
            vertex_updates[batch_index].index_add_(
                0,
                incidence_index[batch_index, :, endpoint_slot],
                endpoint_messages,
            )

    return vertex_updates
