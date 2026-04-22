"""Reference and native op namespaces for pASCNN."""

from __future__ import annotations

from typing import Literal

import torch

from pascnn.native import load_native_extension
from pascnn.ops.reference import _validate_complex_restrict_inputs
from pascnn.ops.reference import _validate_complex_scatter_inputs
from pascnn.ops.reference import _validate_edge_coherence_inputs
from pascnn.ops.reference import _validate_soft_prefix_inputs
from pascnn.ops.reference import complex_restrict_defect as complex_restrict_defect_reference
from pascnn.ops.reference import complex_scatter_add as complex_scatter_add_reference
from pascnn.ops.reference import edge_coherence_uwca as edge_coherence_uwca_reference
from pascnn.ops.reference import soft_prefix_match as soft_prefix_match_reference
from pascnn.types import EdgeCoherenceOutputs
from pascnn.types import EdgeCoherenceParameters
from pascnn.types import PrefixMatchOutputs
from pascnn.types import RestrictDefectOutputs
from pascnn.types import TransportMaps


ComplexScatterBackend = Literal["auto", "reference", "native"]
EdgeCoherenceBackend = Literal["auto", "reference", "native"]
SoftPrefixBackend = Literal["auto", "reference", "native"]
RestrictDefectBackend = Literal["auto", "reference", "native"]


def soft_prefix_match(
    source_digit_probs: torch.Tensor,
    target_digit_probs: torch.Tensor,
    *,
    backend: SoftPrefixBackend = "auto",
) -> PrefixMatchOutputs:
    """Return prefix-match outputs from the selected backend."""

    if backend not in ("auto", "reference", "native"):
        raise ValueError("backend must be one of: auto, reference, native.")

    _validate_soft_prefix_inputs(source_digit_probs, target_digit_probs)

    if backend == "reference":
        return soft_prefix_match_reference(source_digit_probs, target_digit_probs)

    module = load_native_extension(required=(backend == "native"))
    if module is not None and hasattr(torch.ops.pascnn, "soft_prefix_match"):
        outputs = torch.ops.pascnn.soft_prefix_match(
            source_digit_probs,
            target_digit_probs,
        )
        return PrefixMatchOutputs(
            position_agreement=outputs[0],
            prefix_compatibility=outputs[1],
            prefix_depth=outputs[2],
        )

    return soft_prefix_match_reference(source_digit_probs, target_digit_probs)


def complex_restrict_defect(
    source_state: torch.Tensor,
    target_state: torch.Tensor,
    edge_type_index: torch.Tensor,
    transport_maps: TransportMaps,
    *,
    backend: RestrictDefectBackend = "auto",
) -> RestrictDefectOutputs:
    """Return edge transport and defect outputs from the selected backend."""

    if backend not in ("auto", "reference", "native"):
        raise ValueError("backend must be one of: auto, reference, native.")

    _validate_complex_restrict_inputs(
        source_state,
        target_state,
        edge_type_index,
        transport_maps,
    )

    if backend == "reference":
        return complex_restrict_defect_reference(
            source_state,
            target_state,
            edge_type_index,
            transport_maps,
        )

    module = load_native_extension(required=(backend == "native"))
    if module is not None and hasattr(torch.ops.pascnn, "complex_restrict_defect"):
        outputs = torch.ops.pascnn.complex_restrict_defect(
            source_state,
            target_state,
            edge_type_index,
            transport_maps.source_diagonal,
            transport_maps.target_diagonal,
        )
        return RestrictDefectOutputs(
            source_transport=outputs[0],
            target_transport=outputs[1],
            defect=outputs[2],
            defect_norm_sq=outputs[3],
        )

    return complex_restrict_defect_reference(
        source_state,
        target_state,
        edge_type_index,
        transport_maps,
    )


def edge_coherence_uwca(
    source_transport: torch.Tensor,
    target_transport: torch.Tensor,
    prefix_depth: torch.Tensor,
    defect_norm_sq: torch.Tensor,
    edge_type_index: torch.Tensor,
    coherence: EdgeCoherenceParameters,
    *,
    backend: EdgeCoherenceBackend = "auto",
) -> EdgeCoherenceOutputs:
    """Return edge coherence outputs from the selected backend."""

    if backend not in ("auto", "reference", "native"):
        raise ValueError("backend must be one of: auto, reference, native.")

    _validate_edge_coherence_inputs(
        source_transport,
        target_transport,
        prefix_depth,
        defect_norm_sq,
        edge_type_index,
        coherence,
    )

    if backend == "reference":
        return edge_coherence_uwca_reference(
            source_transport,
            target_transport,
            prefix_depth,
            defect_norm_sq,
            edge_type_index,
            coherence,
        )

    module = load_native_extension(required=(backend == "native"))
    if module is not None and hasattr(torch.ops.pascnn, "edge_coherence_uwca"):
        try:
            outputs = torch.ops.pascnn.edge_coherence_uwca(
                source_transport,
                target_transport,
                prefix_depth,
                defect_norm_sq,
                edge_type_index,
                coherence.phase_bias,
                coherence.phase_scale,
                coherence.amplitude_alpha,
                coherence.amplitude_threshold,
                coherence.defect_beta,
            )
        except RuntimeError as exc:
            if backend == "auto" and "not implemented in step 16" in str(exc):
                return edge_coherence_uwca_reference(
                    source_transport,
                    target_transport,
                    prefix_depth,
                    defect_norm_sq,
                    edge_type_index,
                    coherence,
                )
            raise

        return EdgeCoherenceOutputs(
            edge_mean=outputs[0],
            edge_amplitude=outputs[1],
            edge_phase=outputs[2],
            edge_coefficient=outputs[3],
            edge_message=outputs[4],
        )

    return edge_coherence_uwca_reference(
        source_transport,
        target_transport,
        prefix_depth,
        defect_norm_sq,
        edge_type_index,
        coherence,
    )


def complex_scatter_add(
    edge_messages: torch.Tensor,
    incidence_index: torch.Tensor,
    scatter_gains: torch.Tensor | None = None,
    num_vertices: int = 5,
    *,
    backend: ComplexScatterBackend = "auto",
) -> torch.Tensor:
    """Return vertex scatter outputs from the selected backend."""

    if backend not in ("auto", "reference", "native"):
        raise ValueError("backend must be one of: auto, reference, native.")

    _validate_complex_scatter_inputs(
        edge_messages,
        incidence_index,
        scatter_gains,
        num_vertices,
    )

    if backend == "reference":
        return complex_scatter_add_reference(
            edge_messages=edge_messages,
            incidence_index=incidence_index,
            scatter_gains=scatter_gains,
            num_vertices=num_vertices,
        )

    module = load_native_extension(required=(backend == "native"))
    if module is not None and hasattr(torch.ops.pascnn, "complex_scatter_add"):
        try:
            return torch.ops.pascnn.complex_scatter_add(
                edge_messages,
                incidence_index,
                scatter_gains,
                num_vertices,
            )
        except RuntimeError as exc:
            if backend == "auto" and "not implemented in step 16" in str(exc):
                return complex_scatter_add_reference(
                    edge_messages=edge_messages,
                    incidence_index=incidence_index,
                    scatter_gains=scatter_gains,
                    num_vertices=num_vertices,
                )
            raise

    return complex_scatter_add_reference(
        edge_messages=edge_messages,
        incidence_index=incidence_index,
        scatter_gains=scatter_gains,
        num_vertices=num_vertices,
    )


__all__ = [
    "ComplexScatterBackend",
    "EdgeCoherenceBackend",
    "RestrictDefectBackend",
    "SoftPrefixBackend",
    "complex_restrict_defect",
    "complex_scatter_add",
    "edge_coherence_uwca",
    "soft_prefix_match",
    "complex_restrict_defect_reference",
    "complex_scatter_add_reference",
    "edge_coherence_uwca_reference",
    "soft_prefix_match_reference",
]
