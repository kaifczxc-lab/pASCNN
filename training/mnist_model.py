"""Reference end-to-end MNIST model built on top of the generic pASCNN core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.autograd.profiler import record_function

from pascnn.adapters.mnist import MNISTAdapter
from pascnn.adapters.mnist import MNISTDecoderMode
from pascnn.adapters.mnist import MNISTDecodeOutputs
from pascnn.adapters.mnist import MNISTEncoder
from pascnn.adapters.mnist import MNISTLinearAdapter
from pascnn.adapters.mnist import MNIST_NUM_LOGIC_HEADS
from pascnn.core.cell import PASCNNCell
from pascnn.types import CellConfig
from pascnn.types import CellForwardOutputs


@dataclass(frozen=True)
class MNISTModelOutputs:
    embedding: torch.Tensor
    cell: CellForwardOutputs
    decode: MNISTDecodeOutputs


class MNISTReferenceModel(nn.Module):
    """Compose MNIST encoder, generic pASCNN core, and MNIST decoder."""

    def __init__(
        self,
        cell_config: CellConfig,
        encoder_embedding_dim: int = 64,
        encoder_base_channels: int = 16,
        decoder_mode: MNISTDecoderMode = "linear_state",
    ) -> None:
        super().__init__()
        if cell_config.num_logic_heads != MNIST_NUM_LOGIC_HEADS:
            raise ValueError(
                f"MNISTReferenceModel requires num_logic_heads={MNIST_NUM_LOGIC_HEADS}."
            )
        if decoder_mode not in ("linear_state", "codebook"):
            raise ValueError("decoder_mode must be 'linear_state' or 'codebook'.")

        self.cell_config = cell_config
        self.decoder_mode = decoder_mode
        self.encoder = MNISTEncoder(
            embedding_dim=encoder_embedding_dim,
            base_channels=encoder_base_channels,
        )
        self.core = PASCNNCell(cell_config)
        if decoder_mode == "linear_state":
            self.decoder = MNISTLinearAdapter(hidden_dim=cell_config.hidden_dim)
        else:
            self.decoder = MNISTAdapter()

    def forward(self, image: torch.Tensor) -> MNISTModelOutputs:
        with record_function("pascnn.mnist_encoder"):
            embedding = self.encoder(image)
        with record_function("pascnn.core_cell"):
            cell_outputs = self.core(embedding)
        with record_function("pascnn.mnist_adapter"):
            if self.decoder_mode == "linear_state":
                decode_outputs = cast(MNISTLinearAdapter, self.decoder)(cell_outputs)
            else:
                decode_outputs = cast(MNISTAdapter, self.decoder)(cell_outputs.readout)
        return MNISTModelOutputs(
            embedding=embedding,
            cell=cell_outputs,
            decode=decode_outputs,
        )
