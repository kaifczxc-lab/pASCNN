"""Sample-level tree depth-relation OOD benchmark for matched transformer vs pASCNN decoders."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pascnn.adapters.sequence import MaskedMeanSequenceAdapter
from pascnn.adapters.sequence import SequenceBatch
from pascnn.data.code_lm import ByteLevelTokenizer
from pascnn.data.tree_dataset import generate_unique_synthetic_tree_samples
from pascnn.diagnostics import collect_core_diagnostics
from pascnn.models.code_lm import MiniTransformerBackbone
from pascnn.models.code_lm import MiniTransformerConfig
from pascnn.types import CellConfig
from pascnn.types import LOGIC_VERTEX_INDICES
from pascnn.types import LogicalReadoutOutputs
from pascnn.types import OpBackend
from pascnn.core.cell import PASCNNCell


TREE_RELATION_CLASS_NAMES: tuple[str, ...] = (
    "left_deeper",
    "equal_depth",
    "right_deeper",
)
TREE_RELATION_CLASS_CODEBOOK: tuple[tuple[int, ...], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


@dataclass(frozen=True)
class TreeRelationBenchmarkConfig:
    output_dir: str
    train_samples_per_class: int = 128
    val_iid_samples_per_class: int = 64
    val_ood_samples_per_class: int = 64
    test_iid_samples_per_class: int = 64
    test_ood_samples_per_class: int = 64
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 7
    device: str = "auto"
    train_depths: tuple[int, ...] = (2, 3)
    ood_depths: tuple[int, ...] = (4, 5)
    sequence_length: int = 256
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_hidden_dim: int = 128
    dropout: float = 0.0
    num_logic_heads: int = 2
    core_hidden_dim: int = 16
    core_branch_depth: int = 2
    core_branch_base: int = 2
    core_num_layers: int = 1
    core_backend: OpBackend = "reference"


@dataclass(frozen=True)
class TreeRelationBatch:
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_class: torch.Tensor


@dataclass(frozen=True)
class TreeRelationSampleMetadata:
    index: int
    split_name: str
    target_class: int
    target_name: str
    left_depth: int
    right_depth: int
    text: str

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


class TreeRelationDataset(Dataset[TreeRelationBatch]):
    def __init__(
        self,
        *,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: torch.Tensor,
        metadata: tuple[TreeRelationSampleMetadata, ...],
    ) -> None:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [N, T].")
        if attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids.")
        if target_class.ndim != 1 or target_class.shape[0] != token_ids.shape[0]:
            raise ValueError("target_class must have shape [N].")
        if len(metadata) != token_ids.shape[0]:
            raise ValueError("metadata length must match token_ids rows.")
        self.token_ids = token_ids
        self.attention_mask = attention_mask
        self.target_class = target_class
        self.metadata = metadata

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])

    def __getitem__(self, index: int) -> TreeRelationBatch:
        return TreeRelationBatch(
            token_ids=self.token_ids[index],
            attention_mask=self.attention_mask[index],
            target_class=self.target_class[index],
        )


@dataclass(frozen=True)
class TreeRelationDatasetBundle:
    train_dataset: TreeRelationDataset
    val_iid_dataset: TreeRelationDataset
    val_ood_dataset: TreeRelationDataset
    test_iid_dataset: TreeRelationDataset
    test_ood_dataset: TreeRelationDataset
    vocab_size: int


@dataclass(frozen=True)
class TreeRelationClassifierOutputs:
    class_log_scores: torch.Tensor
    class_probabilities: torch.Tensor
    predicted_class: torch.Tensor
    core_diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class TreeRelationEpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_iid_loss: float
    val_iid_accuracy: float
    val_ood_loss: float
    val_ood_accuracy: float
    test_iid_loss: float
    test_iid_accuracy: float
    test_ood_loss: float
    test_ood_accuracy: float
    edge_prefix_depth_to_uniform_ratio_mean: float | None = None
    wave_prefix_depth_to_uniform_ratio_mean: float | None = None
    edge_defect_norm_sq_normalized_mean: float | None = None
    edge_amplitude_mean: float | None = None

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TreeRelationModelRun:
    name: str
    parameter_count: int
    history: tuple[TreeRelationEpochRecord, ...]
    elapsed_seconds: float

    def to_serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": self.parameter_count,
            "history": [record.to_serializable() for record in self.history],
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class TreeRelationBenchmarkOutputs:
    config: TreeRelationBenchmarkConfig
    transformer_run: TreeRelationModelRun
    codebook_run: TreeRelationModelRun
    linear_state_run: TreeRelationModelRun
    report_path: str
    splits_path: str

    def to_serializable(self) -> dict[str, Any]:
        config_dict = asdict(self.config)
        config_dict["output_dir"] = str(config_dict["output_dir"])
        return {
            "config": config_dict,
            "transformer_run": self.transformer_run.to_serializable(),
            "codebook_run": self.codebook_run.to_serializable(),
            "linear_state_run": self.linear_state_run.to_serializable(),
            "report_path": self.report_path,
            "splits_path": self.splits_path,
        }


def _validate_depths(depths: tuple[int, ...], *, split_name: str) -> None:
    if not depths:
        raise ValueError(f"{split_name} must contain at least one depth.")
    if any(depth < 0 for depth in depths):
        raise ValueError(f"{split_name} must be non-negative.")
    if len(set(depths)) != len(depths):
        raise ValueError(f"{split_name} must be unique.")
    if len(depths) < 2:
        raise ValueError(f"{split_name} must contain at least two distinct depths.")


def _validate_config(config: TreeRelationBenchmarkConfig) -> None:
    if config.train_samples_per_class < 1:
        raise ValueError("train_samples_per_class must be at least 1.")
    if config.val_iid_samples_per_class < 1:
        raise ValueError("val_iid_samples_per_class must be at least 1.")
    if config.val_ood_samples_per_class < 1:
        raise ValueError("val_ood_samples_per_class must be at least 1.")
    if config.test_iid_samples_per_class < 1:
        raise ValueError("test_iid_samples_per_class must be at least 1.")
    if config.test_ood_samples_per_class < 1:
        raise ValueError("test_ood_samples_per_class must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if config.eval_batch_size < 1:
        raise ValueError("eval_batch_size must be at least 1.")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if config.sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")
    if config.embedding_dim < 1:
        raise ValueError("embedding_dim must be at least 1.")
    if config.num_heads < 1:
        raise ValueError("num_heads must be at least 1.")
    if config.embedding_dim % config.num_heads != 0:
        raise ValueError("embedding_dim must be divisible by num_heads.")
    if config.num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    if config.ffn_hidden_dim < config.embedding_dim:
        raise ValueError("ffn_hidden_dim must be at least embedding_dim.")
    if config.dropout < 0.0 or config.dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if config.num_logic_heads != len(TREE_RELATION_CLASS_CODEBOOK[0]):
        raise ValueError(
            f"num_logic_heads must be {len(TREE_RELATION_CLASS_CODEBOOK[0])} "
            "for the fixed tree relation codebook."
        )
    if config.core_hidden_dim < 1:
        raise ValueError("core_hidden_dim must be at least 1.")
    if config.core_branch_depth < 1:
        raise ValueError("core_branch_depth must be at least 1.")
    if config.core_branch_base < 2:
        raise ValueError("core_branch_base must be at least 2.")
    if config.core_num_layers < 1:
        raise ValueError("core_num_layers must be at least 1.")
    if config.core_backend not in ("reference", "native", "auto"):
        raise ValueError("core_backend must be one of: reference, native, auto.")
    _validate_depths(config.train_depths, split_name="train_depths")
    _validate_depths(config.ood_depths, split_name="ood_depths")
    if any(depth <= max(config.train_depths) for depth in config.ood_depths):
        raise ValueError("every ood depth must be greater than every train depth.")


def _relation_class_from_depths(left_depth: int, right_depth: int) -> int:
    if left_depth > right_depth:
        return 0
    if left_depth == right_depth:
        return 1
    return 2


def _depth_pairs_for_class(
    depths: tuple[int, ...],
    target_class: int,
) -> tuple[tuple[int, int], ...]:
    pairs = tuple(
        (left_depth, right_depth)
        for left_depth in depths
        for right_depth in depths
        if _relation_class_from_depths(left_depth, right_depth) == target_class
    )
    if not pairs:
        raise ValueError(f"No depth pairs available for class {target_class}.")
    return pairs


def _format_pair_text(left_text: str, right_text: str) -> str:
    return f"L {left_text}\nR {right_text}"


def _generate_unique_tree_text(
    *,
    depth: int,
    rng: random.Random,
    seen_tree_texts: set[str],
) -> str:
    return generate_unique_synthetic_tree_samples(
        depth=depth,
        num_samples=1,
        rng=rng,
        seen_texts=seen_tree_texts,
    )[0].text


def _encode_text_rows(
    *,
    texts: tuple[str, ...],
    sequence_length: int,
    tokenizer: ByteLevelTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_rows: list[list[int]] = []
    mask_rows: list[list[float]] = []
    max_length = 0
    for text in texts:
        token_ids = tokenizer.encode(text, add_bos=False, add_eos=True)
        max_length = max(max_length, len(token_ids))
        if len(token_ids) > sequence_length:
            raise ValueError(
                f"sequence_length={sequence_length} is too small for a tree pair example; "
                f"observed encoded length {len(token_ids)}."
            )
        padding = sequence_length - len(token_ids)
        token_rows.append(token_ids + [tokenizer.pad_token_id] * padding)
        mask_rows.append([1.0] * len(token_ids) + [0.0] * padding)
    return (
        torch.tensor(token_rows, dtype=torch.long),
        torch.tensor(mask_rows, dtype=torch.float32),
    )


def _build_split_dataset(
    *,
    split_name: str,
    samples_per_class: int,
    depths: tuple[int, ...],
    split_seed: int,
    sequence_length: int,
    tokenizer: ByteLevelTokenizer,
    seen_tree_texts: set[str],
) -> TreeRelationDataset:
    rng = random.Random(split_seed)
    token_texts: list[str] = []
    targets: list[int] = []
    metadata_rows: list[TreeRelationSampleMetadata] = []
    depth_pairs_by_class = {
        target_class: _depth_pairs_for_class(depths, target_class)
        for target_class in range(len(TREE_RELATION_CLASS_NAMES))
    }

    for target_class, target_name in enumerate(TREE_RELATION_CLASS_NAMES):
        for _ in range(samples_per_class):
            left_depth, right_depth = rng.choice(depth_pairs_by_class[target_class])
            left_text = _generate_unique_tree_text(
                depth=left_depth,
                rng=rng,
                seen_tree_texts=seen_tree_texts,
            )
            right_text = _generate_unique_tree_text(
                depth=right_depth,
                rng=rng,
                seen_tree_texts=seen_tree_texts,
            )
            pair_text = _format_pair_text(left_text, right_text)
            metadata_rows.append(
                TreeRelationSampleMetadata(
                    index=len(metadata_rows),
                    split_name=split_name,
                    target_class=target_class,
                    target_name=target_name,
                    left_depth=left_depth,
                    right_depth=right_depth,
                    text=pair_text,
                )
            )
            token_texts.append(pair_text)
            targets.append(target_class)

    row_order = list(range(len(metadata_rows)))
    rng.shuffle(row_order)
    shuffled_texts = tuple(token_texts[index] for index in row_order)
    shuffled_targets = torch.tensor(
        [targets[index] for index in row_order],
        dtype=torch.long,
    )
    shuffled_metadata = tuple(metadata_rows[index] for index in row_order)
    token_ids, attention_mask = _encode_text_rows(
        texts=shuffled_texts,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
    )
    return TreeRelationDataset(
        token_ids=token_ids,
        attention_mask=attention_mask,
        target_class=shuffled_targets,
        metadata=shuffled_metadata,
    )


def build_tree_relation_datasets(
    config: TreeRelationBenchmarkConfig,
) -> TreeRelationDatasetBundle:
    _validate_config(config)
    tokenizer = ByteLevelTokenizer()
    seen_tree_texts: set[str] = set()
    return TreeRelationDatasetBundle(
        train_dataset=_build_split_dataset(
            split_name="train",
            samples_per_class=config.train_samples_per_class,
            depths=config.train_depths,
            split_seed=config.seed,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            seen_tree_texts=seen_tree_texts,
        ),
        val_iid_dataset=_build_split_dataset(
            split_name="val_iid",
            samples_per_class=config.val_iid_samples_per_class,
            depths=config.train_depths,
            split_seed=config.seed + 1,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            seen_tree_texts=seen_tree_texts,
        ),
        val_ood_dataset=_build_split_dataset(
            split_name="val_ood",
            samples_per_class=config.val_ood_samples_per_class,
            depths=config.ood_depths,
            split_seed=config.seed + 2,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            seen_tree_texts=seen_tree_texts,
        ),
        test_iid_dataset=_build_split_dataset(
            split_name="test_iid",
            samples_per_class=config.test_iid_samples_per_class,
            depths=config.train_depths,
            split_seed=config.seed + 3,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            seen_tree_texts=seen_tree_texts,
        ),
        test_ood_dataset=_build_split_dataset(
            split_name="test_ood",
            samples_per_class=config.test_ood_samples_per_class,
            depths=config.ood_depths,
            split_seed=config.seed + 4,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            seen_tree_texts=seen_tree_texts,
        ),
        vocab_size=tokenizer.vocab_size,
    )


def _collate_batches(examples: list[TreeRelationBatch]) -> TreeRelationBatch:
    return TreeRelationBatch(
        token_ids=torch.stack([example.token_ids for example in examples], dim=0),
        attention_mask=torch.stack([example.attention_mask for example in examples], dim=0),
        target_class=torch.stack([example.target_class for example in examples], dim=0),
    )


def _make_data_loader(
    dataset: TreeRelationDataset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[TreeRelationBatch]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate_batches,
        drop_last=False,
        persistent_workers=False,
    )


def _move_batch_to_device(
    batch: TreeRelationBatch,
    device: torch.device,
) -> TreeRelationBatch:
    return TreeRelationBatch(
        token_ids=batch.token_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        target_class=batch.target_class.to(device),
    )


class MatchedTransformerTreeRelationClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        config: TreeRelationBenchmarkConfig,
    ) -> None:
        super().__init__()
        backbone_config = MiniTransformerConfig(
            vocab_size=vocab_size,
            max_sequence_length=config.sequence_length,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
        )
        self.encoder = MaskedMeanSequenceAdapter(
            token_backbone=MiniTransformerBackbone(backbone_config),
            embedding_dim=config.embedding_dim,
        )
        self.classifier = nn.Linear(config.embedding_dim, len(TREE_RELATION_CLASS_NAMES))

    def forward(self, batch: TreeRelationBatch) -> TreeRelationClassifierOutputs:
        embedding = self.encoder(
            SequenceBatch(
                token_ids=batch.token_ids,
                attention_mask=batch.attention_mask,
            )
        )
        class_log_scores = self.classifier(embedding)
        class_probabilities = torch.softmax(class_log_scores, dim=-1)
        predicted_class = class_log_scores.argmax(dim=-1)
        return TreeRelationClassifierOutputs(
            class_log_scores=class_log_scores,
            class_probabilities=class_probabilities,
            predicted_class=predicted_class,
        )


class TreeRelationCodebookAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "class_codebook",
            torch.tensor(TREE_RELATION_CLASS_CODEBOOK, dtype=torch.long),
        )
        self._probability_epsilon = torch.finfo(torch.float32).tiny

    def forward(self, readout: LogicalReadoutOutputs) -> TreeRelationClassifierOutputs:
        if not isinstance(readout, LogicalReadoutOutputs):
            raise TypeError("readout must be a LogicalReadoutOutputs instance.")
        logic_probabilities = readout.logic_probabilities
        if logic_probabilities.ndim != 3:
            raise ValueError("logic_probabilities must have shape [B, H, 3].")
        if logic_probabilities.shape[1] != len(TREE_RELATION_CLASS_CODEBOOK[0]):
            raise ValueError("logic_probabilities head count does not match fixed codebook.")
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
        return TreeRelationClassifierOutputs(
            class_log_scores=class_log_scores,
            class_probabilities=class_probabilities,
            predicted_class=predicted_class,
        )


class PASCNNTreeRelationCodebookClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        config: TreeRelationBenchmarkConfig,
    ) -> None:
        super().__init__()
        backbone_config = MiniTransformerConfig(
            vocab_size=vocab_size,
            max_sequence_length=config.sequence_length,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
        )
        self.encoder = MaskedMeanSequenceAdapter(
            token_backbone=MiniTransformerBackbone(backbone_config),
            embedding_dim=config.embedding_dim,
        )
        self.core = PASCNNCell(
            CellConfig(
                hidden_dim=config.core_hidden_dim,
                branch_depth=config.core_branch_depth,
                branch_base=config.core_branch_base,
                num_layers=config.core_num_layers,
                num_logic_heads=config.num_logic_heads,
                op_backend=config.core_backend,
            )
        )
        self.decoder = TreeRelationCodebookAdapter()

    def forward(self, batch: TreeRelationBatch) -> TreeRelationClassifierOutputs:
        embedding = self.encoder(
            SequenceBatch(
                token_ids=batch.token_ids,
                attention_mask=batch.attention_mask,
            )
        )
        cell_outputs = self.core(embedding)
        decode_outputs = self.decoder(cell_outputs.readout)
        diagnostics = collect_core_diagnostics(cell_outputs).as_logging_dict()
        return TreeRelationClassifierOutputs(
            class_log_scores=decode_outputs.class_log_scores,
            class_probabilities=decode_outputs.class_probabilities,
            predicted_class=decode_outputs.predicted_class,
            core_diagnostics=diagnostics,
        )


class PASCNNTreeRelationLinearStateClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        config: TreeRelationBenchmarkConfig,
    ) -> None:
        super().__init__()
        backbone_config = MiniTransformerConfig(
            vocab_size=vocab_size,
            max_sequence_length=config.sequence_length,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
        )
        self.encoder = MaskedMeanSequenceAdapter(
            token_backbone=MiniTransformerBackbone(backbone_config),
            embedding_dim=config.embedding_dim,
        )
        self.core = PASCNNCell(
            CellConfig(
                hidden_dim=config.core_hidden_dim,
                branch_depth=config.core_branch_depth,
                branch_base=config.core_branch_base,
                num_layers=config.core_num_layers,
                num_logic_heads=config.num_logic_heads,
                op_backend=config.core_backend,
            )
        )
        self.classifier = nn.Linear(
            len(LOGIC_VERTEX_INDICES) * config.core_hidden_dim * 2,
            len(TREE_RELATION_CLASS_NAMES),
        )

    def forward(self, batch: TreeRelationBatch) -> TreeRelationClassifierOutputs:
        embedding = self.encoder(
            SequenceBatch(
                token_ids=batch.token_ids,
                attention_mask=batch.attention_mask,
            )
        )
        cell_outputs = self.core(embedding)
        logical_vertex_state = cell_outputs.vertex_state[:, LOGIC_VERTEX_INDICES, :]
        state_features = torch.cat(
            (
                logical_vertex_state.real,
                logical_vertex_state.imag,
            ),
            dim=-1,
        ).reshape(logical_vertex_state.shape[0], -1)
        class_log_scores = self.classifier(state_features)
        class_probabilities = torch.softmax(class_log_scores, dim=-1)
        predicted_class = class_log_scores.argmax(dim=-1)
        diagnostics = collect_core_diagnostics(cell_outputs).as_logging_dict()
        return TreeRelationClassifierOutputs(
            class_log_scores=class_log_scores,
            class_probabilities=class_probabilities,
            predicted_class=predicted_class,
            core_diagnostics=diagnostics,
        )


def _evaluate_split(
    model: nn.Module,
    data_loader: DataLoader[TreeRelationBatch],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    diagnostics_sums: dict[str, float] = {}
    diagnostics_batches = 0
    with torch.inference_mode():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch)
            loss = F.cross_entropy(outputs.class_log_scores, batch.target_class)
            batch_size = batch.target_class.shape[0]
            total_examples += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((outputs.predicted_class == batch.target_class).sum().item())
            if outputs.core_diagnostics is not None:
                diagnostics_batches += 1
                for key, value in outputs.core_diagnostics.items():
                    diagnostics_sums[key] = diagnostics_sums.get(key, 0.0) + float(value)

    metrics = {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }
    if diagnostics_batches > 0:
        metrics.update(
            {key: value / diagnostics_batches for key, value in diagnostics_sums.items()}
        )
    return metrics


def _train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader[TreeRelationBatch],
    optimizer: AdamW,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in data_loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = F.cross_entropy(outputs.class_log_scores, batch.target_class)
        loss.backward()
        optimizer.step()
        batch_size = batch.target_class.shape[0]
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((outputs.predicted_class == batch.target_class).sum().item())
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def _run_model(
    *,
    name: str,
    model: nn.Module,
    optimizer: AdamW,
    train_loader: DataLoader[TreeRelationBatch],
    val_iid_loader: DataLoader[TreeRelationBatch],
    val_ood_loader: DataLoader[TreeRelationBatch],
    test_iid_loader: DataLoader[TreeRelationBatch],
    test_ood_loader: DataLoader[TreeRelationBatch],
    device: torch.device,
    epochs: int,
) -> TreeRelationModelRun:
    history: list[TreeRelationEpochRecord] = []
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _train_one_epoch(model, train_loader, optimizer, device)
        val_iid_metrics = _evaluate_split(model, val_iid_loader, device)
        val_ood_metrics = _evaluate_split(model, val_ood_loader, device)
        test_iid_metrics = _evaluate_split(model, test_iid_loader, device)
        test_ood_metrics = _evaluate_split(model, test_ood_loader, device)
        history.append(
            TreeRelationEpochRecord(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                val_iid_loss=val_iid_metrics["loss"],
                val_iid_accuracy=val_iid_metrics["accuracy"],
                val_ood_loss=val_ood_metrics["loss"],
                val_ood_accuracy=val_ood_metrics["accuracy"],
                test_iid_loss=test_iid_metrics["loss"],
                test_iid_accuracy=test_iid_metrics["accuracy"],
                test_ood_loss=test_ood_metrics["loss"],
                test_ood_accuracy=test_ood_metrics["accuracy"],
                edge_prefix_depth_to_uniform_ratio_mean=test_ood_metrics.get(
                    "edge_prefix_depth_to_uniform_ratio_mean"
                ),
                wave_prefix_depth_to_uniform_ratio_mean=test_ood_metrics.get(
                    "wave_prefix_depth_to_uniform_ratio_mean"
                ),
                edge_defect_norm_sq_normalized_mean=test_ood_metrics.get(
                    "edge_defect_norm_sq_normalized_mean"
                ),
                edge_amplitude_mean=test_ood_metrics.get("edge_amplitude_mean"),
            )
        )
    return TreeRelationModelRun(
        name=name,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        history=tuple(history),
        elapsed_seconds=time.perf_counter() - start_time,
    )


def _write_split_manifest(
    *,
    bundle: TreeRelationDatasetBundle,
    output_path: Path,
) -> None:
    payload = {
        "train": [metadata.to_serializable() for metadata in bundle.train_dataset.metadata],
        "val_iid": [metadata.to_serializable() for metadata in bundle.val_iid_dataset.metadata],
        "val_ood": [metadata.to_serializable() for metadata in bundle.val_ood_dataset.metadata],
        "test_iid": [metadata.to_serializable() for metadata in bundle.test_iid_dataset.metadata],
        "test_ood": [metadata.to_serializable() for metadata in bundle.test_ood_dataset.metadata],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_tree_relation_benchmark(
    config: TreeRelationBenchmarkConfig,
    datasets_bundle: TreeRelationDatasetBundle | None = None,
) -> TreeRelationBenchmarkOutputs:
    _validate_config(config)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "tree_relation_benchmark.json"
    splits_path = output_dir / "tree_relation_splits.json"

    if datasets_bundle is None:
        datasets_bundle = build_tree_relation_datasets(config)
    _write_split_manifest(bundle=datasets_bundle, output_path=splits_path)

    device = _resolve_device(config.device)
    train_loader = _make_data_loader(
        datasets_bundle.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_iid_loader = _make_data_loader(
        datasets_bundle.val_iid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
    )
    val_ood_loader = _make_data_loader(
        datasets_bundle.val_ood_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
    )
    test_iid_loader = _make_data_loader(
        datasets_bundle.test_iid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
    )
    test_ood_loader = _make_data_loader(
        datasets_bundle.test_ood_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    transformer_model = MatchedTransformerTreeRelationClassifier(
        vocab_size=datasets_bundle.vocab_size,
        config=config,
    ).to(device)
    transformer_optimizer = AdamW(
        transformer_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    codebook_model = PASCNNTreeRelationCodebookClassifier(
        vocab_size=datasets_bundle.vocab_size,
        config=config,
    ).to(device)
    codebook_optimizer = AdamW(
        codebook_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    linear_state_model = PASCNNTreeRelationLinearStateClassifier(
        vocab_size=datasets_bundle.vocab_size,
        config=config,
    ).to(device)
    linear_state_optimizer = AdamW(
        linear_state_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    transformer_run = _run_model(
        name="matched_transformer_meanpool_tree_relation_classifier",
        model=transformer_model,
        optimizer=transformer_optimizer,
        train_loader=train_loader,
        val_iid_loader=val_iid_loader,
        val_ood_loader=val_ood_loader,
        test_iid_loader=test_iid_loader,
        test_ood_loader=test_ood_loader,
        device=device,
        epochs=config.epochs,
    )
    codebook_run = _run_model(
        name="pascnn_tree_relation_codebook_classifier",
        model=codebook_model,
        optimizer=codebook_optimizer,
        train_loader=train_loader,
        val_iid_loader=val_iid_loader,
        val_ood_loader=val_ood_loader,
        test_iid_loader=test_iid_loader,
        test_ood_loader=test_ood_loader,
        device=device,
        epochs=config.epochs,
    )
    linear_state_run = _run_model(
        name="pascnn_tree_relation_linear_state_classifier",
        model=linear_state_model,
        optimizer=linear_state_optimizer,
        train_loader=train_loader,
        val_iid_loader=val_iid_loader,
        val_ood_loader=val_ood_loader,
        test_iid_loader=test_iid_loader,
        test_ood_loader=test_ood_loader,
        device=device,
        epochs=config.epochs,
    )

    outputs = TreeRelationBenchmarkOutputs(
        config=config,
        transformer_run=transformer_run,
        codebook_run=codebook_run,
        linear_state_run=linear_state_run,
        report_path=str(report_path),
        splits_path=str(splits_path),
    )
    report_path.write_text(
        json.dumps(outputs.to_serializable(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return outputs
