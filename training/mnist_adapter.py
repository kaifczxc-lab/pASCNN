"""MNIST encoder and decoder adapters for the generic pASCNN core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from pascnn.types import CellForwardOutputs
from pascnn.types import LOGIC_VERTEX_INDICES
from pascnn.types import LogicalReadoutOutputs


MNIST_NUM_CLASSES = 10
MNIST_NUM_LOGIC_HEADS = 5
MNIST_IMAGE_HEIGHT = 28
MNIST_IMAGE_WIDTH = 28
MNIST_CLASS_CODEBOOK: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2),
    (0, 0, 1, 1, 2),
    (1, 1, 0, 2, 0),
    (0, 2, 2, 0, 1),
    (2, 0, 0, 1, 1),
    (1, 1, 2, 0, 2),
    (0, 2, 1, 2, 0),
    (2, 1, 1, 0, 0),
)
MNISTDecoderMode = Literal["linear_state", "codebook"]


@dataclass(frozen=True)
class MNISTDecodeOutputs:
    class_log_scores: torch.Tensor
    class_probabilities: torch.Tensor
    predicted_class: torch.Tensor
    predicted_codeword: torch.Tensor


class MNISTEncoder(nn.Module):
    """Small image encoder that maps MNIST images to core embeddings."""

    def __init__(
        self,
        embedding_dim: int = 64,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")
        if base_channels < 1:
            raise ValueError("base_channels must be at least 1.")

        hidden_channels = base_channels * 2
        output_channels = base_channels * 4
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_channels * 7 * 7, embedding_dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(1)
        if image.ndim != 4:
            raise ValueError("image must have shape [B, 1, 28, 28] or [B, 28, 28].")
        if image.shape[1] != 1:
            raise ValueError("MNISTEncoder expects a single input channel.")
        if image.shape[2] != MNIST_IMAGE_HEIGHT or image.shape[3] != MNIST_IMAGE_WIDTH:
            raise ValueError("MNISTEncoder expects spatial shape 28x28.")
        if not image.is_floating_point() or image.is_complex():
            raise TypeError("image must be a real floating tensor.")

        feature_map = self.features(image)
        return self.projection(feature_map)


class MNISTAdapter(nn.Module):
    """Decode multi-head ternary readout into ten MNIST classes."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "class_codebook",
            torch.tensor(MNIST_CLASS_CODEBOOK, dtype=torch.long),
        )
        self._probability_epsilon = torch.finfo(torch.float32).tiny

    def forward(self, readout: LogicalReadoutOutputs) -> MNISTDecodeOutputs:
        if not isinstance(readout, LogicalReadoutOutputs):
            raise TypeError("readout must be a LogicalReadoutOutputs instance.")

        logic_probabilities = readout.logic_probabilities
        if logic_probabilities.ndim != 3:
            raise ValueError("logic_probabilities must have shape [B, H, 3].")
        if logic_probabilities.shape[1] != MNIST_NUM_LOGIC_HEADS:
            raise ValueError(
                f"MNISTAdapter requires exactly {MNIST_NUM_LOGIC_HEADS} ternary heads."
            )
        if logic_probabilities.shape[2] != 3:
            raise ValueError("logic_probabilities must have a ternary final dimension.")
        if not logic_probabilities.is_floating_point() or logic_probabilities.is_complex():
            raise TypeError("logic_probabilities must be a real floating tensor.")

        probabilities = logic_probabilities.clamp_min(self._probability_epsilon)
        log_probabilities = probabilities.log()

        batch_size = log_probabilities.shape[0]
        num_classes = self.class_codebook.shape[0]
        codebook = self.class_codebook.to(device=log_probabilities.device)

        expanded_log_probabilities = log_probabilities.unsqueeze(1).expand(
            batch_size,
            num_classes,
            -1,
            -1,
        )
        expanded_codebook = codebook.unsqueeze(0).expand(batch_size, -1, -1)
        class_log_scores = torch.gather(
            expanded_log_probabilities,
            dim=-1,
            index=expanded_codebook.unsqueeze(-1),
        ).squeeze(-1).sum(dim=-1)
        class_probabilities = torch.softmax(class_log_scores, dim=-1)
        predicted_class = class_log_scores.argmax(dim=-1)
        predicted_codeword = codebook.index_select(0, predicted_class)

        return MNISTDecodeOutputs(
            class_log_scores=class_log_scores,
            class_probabilities=class_probabilities,
            predicted_class=predicted_class,
            predicted_codeword=predicted_codeword,
        )


class MNISTLinearAdapter(nn.Module):
    """Decode logical vertex states to MNIST classes with a linear state head."""

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1.")
        self.hidden_dim = hidden_dim
        self.feature_dim = len(LOGIC_VERTEX_INDICES) * hidden_dim * 2
        self.feature_scale = float(self.feature_dim)
        self.classifier = nn.Linear(
            self.feature_dim,
            MNIST_NUM_CLASSES,
        )
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, cell_outputs: CellForwardOutputs) -> MNISTDecodeOutputs:
        if not isinstance(cell_outputs, CellForwardOutputs):
            raise TypeError("cell_outputs must be a CellForwardOutputs instance.")

        vertex_state = cell_outputs.vertex_state
        if vertex_state.ndim != 3:
            raise ValueError("vertex_state must have shape [B, V, D].")
        if vertex_state.shape[1] <= max(LOGIC_VERTEX_INDICES):
            raise ValueError("vertex_state does not contain all logical vertices.")
        if vertex_state.shape[2] != self.hidden_dim:
            raise ValueError("vertex_state hidden dim does not match adapter hidden_dim.")
        if not vertex_state.is_complex():
            raise TypeError("vertex_state must be a complex tensor.")

        logic_probabilities = cell_outputs.readout.logic_probabilities
        if logic_probabilities.ndim != 3:
            raise ValueError("logic_probabilities must have shape [B, H, 3].")
        if logic_probabilities.shape[1] != MNIST_NUM_LOGIC_HEADS:
            raise ValueError(
                f"MNISTLinearAdapter requires exactly {MNIST_NUM_LOGIC_HEADS} ternary heads."
            )

        logical_vertex_state = vertex_state[:, LOGIC_VERTEX_INDICES, :]
        state_features = torch.cat(
            (
                logical_vertex_state.real,
                logical_vertex_state.imag,
            ),
            dim=-1,
        ).reshape(logical_vertex_state.shape[0], -1) / self.feature_scale
        class_log_scores = self.classifier(state_features)
        class_probabilities = torch.softmax(class_log_scores, dim=-1)
        predicted_class = class_log_scores.argmax(dim=-1)
        predicted_codeword = logic_probabilities.argmax(dim=-1).to(dtype=torch.long)

        return MNISTDecodeOutputs(
            class_log_scores=class_log_scores,
            class_probabilities=class_probabilities,
            predicted_class=predicted_class,
            predicted_codeword=predicted_codeword,
        )
