"""Synthetic left-right relation OOD benchmark for transformer vs pASCNN decoders."""

from __future__ import annotations

import csv
from dataclasses import asdict
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pascnn.core.cell import PASCNNCell
from pascnn.diagnostics import collect_core_diagnostics
from pascnn.training.cifar10_benchmark import PatchTransformerImageEncoder
from pascnn.types import CellConfig
from pascnn.types import LOGIC_VERTEX_INDICES
from pascnn.types import LogicalReadoutOutputs
from pascnn.types import OpBackend


RELATION_IMAGE_SIZE = 32
RELATION_IMAGE_CHANNELS = 3
RELATION_NUM_CLASSES = 4
RELATION_NUM_SHAPES = 3
RELATION_NUM_COLORS = 3
RELATION_NUM_SIZE_BINS = 2
RELATION_NUM_Y_BINS = 3
RELATION_CLASS_CODEBOOK: tuple[tuple[int, ...], ...] = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
)
RELATION_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 0),  # same
    (0, 1),  # color shift
    (1, 0),  # shape shift
    (1, 1),  # both shift
)
ALL_RELATION_COMBOS: tuple[tuple[int, int, int, int], ...] = tuple(
    tuple(int(component) for component in combo)
    for combo in itertools.product(
        range(RELATION_NUM_SHAPES),
        range(RELATION_NUM_COLORS),
        range(RELATION_NUM_SIZE_BINS),
        range(RELATION_NUM_Y_BINS),
    )
)
TRAIN_COMBOS: tuple[tuple[int, int, int, int], ...] = tuple(
    combo
    for combo in ALL_RELATION_COMBOS
    if (combo[0] + 2 * combo[1] + combo[2] + combo[3]) % 3 != 0
)
ALL_OOD_COMBOS: tuple[tuple[int, int, int, int], ...] = tuple(
    combo
    for combo in ALL_RELATION_COMBOS
    if combo not in TRAIN_COMBOS
)
RELATION_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.20, 0.20),
    (0.20, 0.90, 0.30),
    (0.20, 0.45, 0.95),
)


@dataclass(frozen=True)
class LeftRightRelationBenchmarkConfig:
    output_dir: str
    train_samples: int = 4000
    val_samples: int = 1000
    val_ood_samples: int = 1000
    test_iid_samples: int = 1000
    test_ood_samples: int = 1000
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    core_learning_rate: float = 1e-3
    core_coherence_learning_rate: float = 3e-3
    core_weight_decay: float = 0.0
    warmup_steps: int = 20
    gradient_clip_norm: float = 1.0
    device: str = "auto"
    seed: int = 7
    patch_size: int = 4
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
class LeftRightRelationDatasetBundle:
    train_dataset: Dataset[Any]
    val_dataset: Dataset[Any]
    val_ood_dataset: Dataset[Any]
    test_iid_dataset: Dataset[Any]
    test_ood_dataset: Dataset[Any]


@dataclass(frozen=True)
class RelationClassifierOutputs:
    class_log_scores: torch.Tensor
    class_probabilities: torch.Tensor
    predicted_class: torch.Tensor
    core_diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class RelationSampleMetadata:
    index: int
    combo_index: int
    relation_index: int
    left_shape: int
    left_color: int
    size_bin: int
    y_bin: int
    shape_shift: int
    color_shift: int
    right_shape: int
    right_color: int
    radius: int
    left_center_x: int
    left_center_y: int
    right_center_x: int
    right_center_y: int


@dataclass(frozen=True)
class RelationEpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
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
class RelationModelRun:
    name: str
    parameter_count: int
    history: tuple[RelationEpochRecord, ...]
    elapsed_seconds: float
    selected_epoch: int
    selected_val_ood_accuracy: float
    selected_test_iid_accuracy: float
    selected_test_ood_accuracy: float
    selected_checkpoint_path: str
    selected_prediction_csv_paths: dict[str, str]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": self.parameter_count,
            "history": [record.to_serializable() for record in self.history],
            "elapsed_seconds": self.elapsed_seconds,
            "selected_epoch": self.selected_epoch,
            "selected_val_ood_accuracy": self.selected_val_ood_accuracy,
            "selected_test_iid_accuracy": self.selected_test_iid_accuracy,
            "selected_test_ood_accuracy": self.selected_test_ood_accuracy,
            "selected_checkpoint_path": self.selected_checkpoint_path,
            "selected_prediction_csv_paths": dict(self.selected_prediction_csv_paths),
        }


@dataclass(frozen=True)
class LeftRightRelationBenchmarkOutputs:
    config: LeftRightRelationBenchmarkConfig
    transformer_run: RelationModelRun
    codebook_run: RelationModelRun
    linear_state_run: RelationModelRun
    report_path: str

    def to_serializable(self) -> dict[str, Any]:
        config_dict = asdict(self.config)
        config_dict["output_dir"] = str(config_dict["output_dir"])
        return {
            "config": config_dict,
            "transformer_run": self.transformer_run.to_serializable(),
            "codebook_run": self.codebook_run.to_serializable(),
            "linear_state_run": self.linear_state_run.to_serializable(),
            "report_path": self.report_path,
        }


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _validate_config(config: LeftRightRelationBenchmarkConfig) -> None:
    if config.train_samples < 1:
        raise ValueError("train_samples must be at least 1.")
    if config.val_samples < 1:
        raise ValueError("val_samples must be at least 1.")
    if config.val_ood_samples < 1:
        raise ValueError("val_ood_samples must be at least 1.")
    if config.test_iid_samples < 1:
        raise ValueError("test_iid_samples must be at least 1.")
    if config.test_ood_samples < 1:
        raise ValueError("test_ood_samples must be at least 1.")
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
    if config.core_learning_rate <= 0.0:
        raise ValueError("core_learning_rate must be positive.")
    if config.core_coherence_learning_rate <= 0.0:
        raise ValueError("core_coherence_learning_rate must be positive.")
    if config.core_weight_decay < 0.0:
        raise ValueError("core_weight_decay must be non-negative.")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if config.gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive.")
    if config.patch_size < 1 or RELATION_IMAGE_SIZE % config.patch_size != 0:
        raise ValueError("patch_size must divide 32.")
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
    if config.num_logic_heads != len(RELATION_CLASS_CODEBOOK[0]):
        raise ValueError(
            f"num_logic_heads must be {len(RELATION_CLASS_CODEBOOK[0])} for the fixed relation codebook."
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


def _split_ood_combo_pool(
    seed: int,
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int, int, int], ...]]:
    combo_list = list(ALL_OOD_COMBOS)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(combo_list), generator=generator).tolist()
    shuffled = [combo_list[index] for index in permutation]
    midpoint = len(shuffled) // 2
    if midpoint < 1 or midpoint >= len(shuffled):
        raise ValueError("ALL_OOD_COMBOS must be splittable into non-empty val/test pools.")
    return tuple(shuffled[:midpoint]), tuple(shuffled[midpoint:])


def _disk_mask(
    image_size: int,
    *,
    center_x: int,
    center_y: int,
    radius: int,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(image_size),
        torch.arange(image_size),
        indexing="ij",
    )
    return ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= radius ** 2


def _square_mask(
    image_size: int,
    *,
    center_x: int,
    center_y: int,
    radius: int,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(image_size),
        torch.arange(image_size),
        indexing="ij",
    )
    return ((xx - center_x).abs() <= radius) & ((yy - center_y).abs() <= radius)


def _triangle_mask(
    image_size: int,
    *,
    center_x: int,
    center_y: int,
    radius: int,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(image_size),
        torch.arange(image_size),
        indexing="ij",
    )
    top_y = center_y - radius
    bottom_y = center_y + radius
    vertical = (yy >= top_y) & (yy <= bottom_y)
    height = torch.clamp(yy - top_y, min=0)
    max_half_width = ((height.float() / float(max(1, 2 * radius))) * radius).floor()
    horizontal = (xx - center_x).abs() <= max_half_width
    return vertical & horizontal


def _shape_mask(
    shape_index: int,
    *,
    image_size: int,
    center_x: int,
    center_y: int,
    radius: int,
) -> torch.Tensor:
    if shape_index == 0:
        return _square_mask(
            image_size,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
        )
    if shape_index == 1:
        return _disk_mask(
            image_size,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
        )
    if shape_index == 2:
        return _triangle_mask(
            image_size,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
        )
    raise ValueError("shape_index must be 0, 1, or 2.")


def _draw_shape(
    image: torch.Tensor,
    *,
    shape_index: int,
    color_index: int,
    center_x: int,
    center_y: int,
    radius: int,
) -> None:
    mask = _shape_mask(
        shape_index,
        image_size=image.shape[-1],
        center_x=center_x,
        center_y=center_y,
        radius=radius,
    )
    color = image.new_tensor(RELATION_COLORS[color_index]).view(3, 1)
    image[:, mask] = color.expand(-1, int(mask.sum().item()))


class LeftRightRelationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        num_samples: int,
        combo_pool: tuple[tuple[int, int, int, int], ...],
        seed: int,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        if len(combo_pool) < 1:
            raise ValueError("combo_pool must be non-empty.")
        self.num_samples = num_samples
        self.combo_pool = combo_pool
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def sample_metadata(self, index: int) -> RelationSampleMetadata:
        generator = torch.Generator().manual_seed(self.seed + index)
        relation_index = int(
            torch.randint(0, RELATION_NUM_CLASSES, (1,), generator=generator).item()
        )
        combo_index = int(
            torch.randint(0, len(self.combo_pool), (1,), generator=generator).item()
        )
        left_shape, left_color, size_bin, y_bin = self.combo_pool[combo_index]
        shape_shift, color_shift = RELATION_OFFSETS[relation_index]
        right_shape = (left_shape + shape_shift) % RELATION_NUM_SHAPES
        right_color = (left_color + color_shift) % RELATION_NUM_COLORS
        radius = 4 if size_bin == 0 else 6
        left_center_x = int(torch.randint(7, 11, (1,), generator=generator).item())
        right_center_x = int(torch.randint(21, 25, (1,), generator=generator).item())
        y_band_centers = (9, 16, 23)
        base_center_y = y_band_centers[y_bin]
        left_center_y = base_center_y + int(
            torch.randint(-1, 2, (1,), generator=generator).item()
        )
        right_center_y = base_center_y + int(
            torch.randint(-1, 2, (1,), generator=generator).item()
        )
        return RelationSampleMetadata(
            index=index,
            combo_index=combo_index,
            relation_index=relation_index,
            left_shape=left_shape,
            left_color=left_color,
            size_bin=size_bin,
            y_bin=y_bin,
            shape_shift=shape_shift,
            color_shift=color_shift,
            right_shape=right_shape,
            right_color=right_color,
            radius=radius,
            left_center_x=left_center_x,
            left_center_y=left_center_y,
            right_center_x=right_center_x,
            right_center_y=right_center_y,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = self.sample_metadata(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        image = 0.02 * torch.rand(
            RELATION_IMAGE_CHANNELS,
            RELATION_IMAGE_SIZE,
            RELATION_IMAGE_SIZE,
            generator=generator,
        )
        _draw_shape(
            image,
            shape_index=metadata.left_shape,
            color_index=metadata.left_color,
            center_x=metadata.left_center_x,
            center_y=metadata.left_center_y,
            radius=metadata.radius,
        )
        _draw_shape(
            image,
            shape_index=metadata.right_shape,
            color_index=metadata.right_color,
            center_x=metadata.right_center_x,
            center_y=metadata.right_center_y,
            radius=metadata.radius,
        )
        image = image.clamp_(0.0, 1.0)
        return image, torch.tensor(metadata.relation_index, dtype=torch.long)


def build_left_right_relation_datasets(
    config: LeftRightRelationBenchmarkConfig,
) -> LeftRightRelationDatasetBundle:
    _validate_config(config)
    val_ood_combos, test_ood_combos = _split_ood_combo_pool(config.seed)
    return LeftRightRelationDatasetBundle(
        train_dataset=LeftRightRelationDataset(
            num_samples=config.train_samples,
            combo_pool=TRAIN_COMBOS,
            seed=config.seed,
        ),
        val_dataset=LeftRightRelationDataset(
            num_samples=config.val_samples,
            combo_pool=TRAIN_COMBOS,
            seed=config.seed + 10_000,
        ),
        val_ood_dataset=LeftRightRelationDataset(
            num_samples=config.val_ood_samples,
            combo_pool=val_ood_combos,
            seed=config.seed + 15_000,
        ),
        test_iid_dataset=LeftRightRelationDataset(
            num_samples=config.test_iid_samples,
            combo_pool=TRAIN_COMBOS,
            seed=config.seed + 20_000,
        ),
        test_ood_dataset=LeftRightRelationDataset(
            num_samples=config.test_ood_samples,
            combo_pool=test_ood_combos,
            seed=config.seed + 30_000,
        ),
    )


def _make_data_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )


def _move_batch_to_device(
    image: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return image.to(device), target.to(device)


class TransformerRelationClassifier(nn.Module):
    def __init__(self, *, config: LeftRightRelationBenchmarkConfig) -> None:
        super().__init__()
        self.encoder = PatchTransformerImageEncoder(
            patch_size=config.patch_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
        )
        self.classifier = nn.Linear(config.embedding_dim, RELATION_NUM_CLASSES)

    def forward(self, image: torch.Tensor) -> RelationClassifierOutputs:
        embedding = self.encoder(image)
        class_log_scores = self.classifier(embedding)
        class_probabilities = torch.softmax(class_log_scores, dim=-1)
        predicted_class = class_log_scores.argmax(dim=-1)
        return RelationClassifierOutputs(
            class_log_scores=class_log_scores,
            class_probabilities=class_probabilities,
            predicted_class=predicted_class,
        )


class RelationCodebookAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "class_codebook",
            torch.tensor(RELATION_CLASS_CODEBOOK, dtype=torch.long),
        )
        self._probability_epsilon = 1e-8

    def forward(self, readout: LogicalReadoutOutputs) -> RelationClassifierOutputs:
        if not isinstance(readout, LogicalReadoutOutputs):
            raise TypeError("readout must be a LogicalReadoutOutputs instance.")
        logic_probabilities = readout.logic_probabilities
        if logic_probabilities.ndim != 3:
            raise ValueError("logic_probabilities must have shape [B, H, 3].")
        if logic_probabilities.shape[1] != len(RELATION_CLASS_CODEBOOK[0]):
            raise ValueError("logic_probabilities head count does not match relation codebook.")
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
        return RelationClassifierOutputs(
            class_log_scores=class_log_scores,
            class_probabilities=class_probabilities,
            predicted_class=predicted_class,
        )


class PASCNNRelationCodebookClassifier(nn.Module):
    def __init__(self, *, config: LeftRightRelationBenchmarkConfig) -> None:
        super().__init__()
        self.encoder = PatchTransformerImageEncoder(
            patch_size=config.patch_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
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
        self.decoder = RelationCodebookAdapter()

    def forward(self, image: torch.Tensor) -> RelationClassifierOutputs:
        embedding = self.encoder(image)
        cell_outputs = self.core(embedding)
        decode_outputs = self.decoder(cell_outputs.readout)
        diagnostics = collect_core_diagnostics(cell_outputs).as_logging_dict()
        return RelationClassifierOutputs(
            class_log_scores=decode_outputs.class_log_scores,
            class_probabilities=decode_outputs.class_probabilities,
            predicted_class=decode_outputs.predicted_class,
            core_diagnostics=diagnostics,
        )


class PASCNNRelationLinearStateClassifier(nn.Module):
    def __init__(self, *, config: LeftRightRelationBenchmarkConfig) -> None:
        super().__init__()
        self.encoder = PatchTransformerImageEncoder(
            patch_size=config.patch_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
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
            RELATION_NUM_CLASSES,
        )

    def forward(self, image: torch.Tensor) -> RelationClassifierOutputs:
        embedding = self.encoder(image)
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
        return RelationClassifierOutputs(
            class_log_scores=class_log_scores,
            class_probabilities=class_probabilities,
            predicted_class=predicted_class,
            core_diagnostics=diagnostics,
        )


def _evaluate_split(
    model: nn.Module,
    data_loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    diagnostics_sums: dict[str, float] = {}
    diagnostics_batches = 0
    with torch.inference_mode():
        for image, target in data_loader:
            image, target = _move_batch_to_device(image, target, device)
            outputs = model(image)
            loss = F.cross_entropy(outputs.class_log_scores, target)
            batch_size = target.shape[0]
            total_examples += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((outputs.predicted_class == target).sum().item())
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


def _relation_name(relation_index: int) -> str:
    relation_names = ("same", "color_shift", "shape_shift", "both_shift")
    return relation_names[relation_index]


def _export_prediction_csv(
    *,
    model: nn.Module,
    dataset: LeftRightRelationDataset,
    split_name: str,
    device: torch.device,
    output_path: Path,
    batch_size: int,
) -> None:
    model.eval()
    fieldnames = [
        "split",
        "index",
        "combo_index",
        "relation_index",
        "relation_name",
        "left_shape",
        "left_color",
        "right_shape",
        "right_color",
        "size_bin",
        "y_bin",
        "shape_shift",
        "color_shift",
        "radius",
        "left_center_x",
        "left_center_y",
        "right_center_x",
        "right_center_y",
        "target_class",
        "predicted_class",
        "correct",
        "confidence",
        "probability_class_0",
        "probability_class_1",
        "probability_class_2",
        "probability_class_3",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with torch.inference_mode():
            for start_index in range(0, len(dataset), batch_size):
                batch_indices = range(start_index, min(start_index + batch_size, len(dataset)))
                batch_images: list[torch.Tensor] = []
                batch_targets: list[int] = []
                batch_metadata: list[RelationSampleMetadata] = []
                for sample_index in batch_indices:
                    image, target = dataset[sample_index]
                    batch_images.append(image)
                    batch_targets.append(int(target.item()))
                    batch_metadata.append(dataset.sample_metadata(sample_index))
                image_batch = torch.stack(batch_images, dim=0).to(device)
                outputs = model(image_batch)
                probabilities = outputs.class_probabilities.detach().cpu()
                predicted = outputs.predicted_class.detach().cpu()
                for row_index, metadata in enumerate(batch_metadata):
                    probability_values = probabilities[row_index].tolist()
                    predicted_class = int(predicted[row_index].item())
                    target_class = batch_targets[row_index]
                    writer.writerow(
                        {
                            "split": split_name,
                            "index": metadata.index,
                            "combo_index": metadata.combo_index,
                            "relation_index": metadata.relation_index,
                            "relation_name": _relation_name(metadata.relation_index),
                            "left_shape": metadata.left_shape,
                            "left_color": metadata.left_color,
                            "right_shape": metadata.right_shape,
                            "right_color": metadata.right_color,
                            "size_bin": metadata.size_bin,
                            "y_bin": metadata.y_bin,
                            "shape_shift": metadata.shape_shift,
                            "color_shift": metadata.color_shift,
                            "radius": metadata.radius,
                            "left_center_x": metadata.left_center_x,
                            "left_center_y": metadata.left_center_y,
                            "right_center_x": metadata.right_center_x,
                            "right_center_y": metadata.right_center_y,
                            "target_class": target_class,
                            "predicted_class": predicted_class,
                            "correct": int(predicted_class == target_class),
                            "confidence": max(probability_values),
                            "probability_class_0": probability_values[0],
                            "probability_class_1": probability_values[1],
                            "probability_class_2": probability_values[2],
                            "probability_class_3": probability_values[3],
                        }
                    )


def _train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader[Any],
    optimizer: Optimizer,
    device: torch.device,
    *,
    warmup_base_lrs: tuple[float, ...],
    warmup_steps: int,
    global_step: int,
    gradient_clip_norm: float,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for image, target in data_loader:
        image, target = _move_batch_to_device(image, target, device)
        if warmup_steps > 0 and global_step < warmup_steps:
            warmup_scale = float(global_step + 1) / float(warmup_steps)
            for param_group, base_lr in zip(
                optimizer.param_groups,
                warmup_base_lrs,
                strict=True,
            ):
                param_group["lr"] = base_lr * warmup_scale
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        loss = F.cross_entropy(outputs.class_log_scores, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        global_step += 1
        batch_size = target.shape[0]
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((outputs.predicted_class == target).sum().item())
    return (
        {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        },
        global_step,
    )


def _build_optimizer(
    model: nn.Module,
    config: LeftRightRelationBenchmarkConfig,
) -> tuple[Optimizer, tuple[float, ...]]:
    if not hasattr(model, "core"):
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return optimizer, (config.learning_rate,)

    coherence_parameter_names = {
        "core.phase_bias",
        "core.phase_scale",
        "core.amplitude_alpha",
        "core.amplitude_threshold",
        "core.defect_beta",
    }
    encoder_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    core_parameters: list[nn.Parameter] = []
    coherence_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_parameters.append(parameter)
        elif name in coherence_parameter_names:
            coherence_parameters.append(parameter)
        elif name.startswith("core."):
            core_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    base_lrs: list[float] = []
    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
        )
        base_lrs.append(config.learning_rate)
    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
        )
        base_lrs.append(config.learning_rate)
    if core_parameters:
        parameter_groups.append(
            {
                "params": core_parameters,
                "lr": config.core_learning_rate,
                "weight_decay": config.core_weight_decay,
            }
        )
        base_lrs.append(config.core_learning_rate)
    if coherence_parameters:
        parameter_groups.append(
            {
                "params": coherence_parameters,
                "lr": config.core_coherence_learning_rate,
                "weight_decay": 0.0,
            }
        )
        base_lrs.append(config.core_coherence_learning_rate)

    optimizer = AdamW(parameter_groups)
    return optimizer, tuple(base_lrs)


def _run_model(
    *,
    name: str,
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    val_ood_loader: DataLoader[Any],
    test_iid_loader: DataLoader[Any],
    test_ood_loader: DataLoader[Any],
    val_ood_dataset: LeftRightRelationDataset,
    test_iid_dataset: LeftRightRelationDataset,
    test_ood_dataset: LeftRightRelationDataset,
    config: LeftRightRelationBenchmarkConfig,
    device: torch.device,
    output_dir: Path,
) -> RelationModelRun:
    first_image, _ = next(iter(train_loader))
    first_image = first_image.to(device)
    _ = model(first_image)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer, warmup_base_lrs = _build_optimizer(model, config)
    history: list[RelationEpochRecord] = []
    start_time = time.perf_counter()
    global_step = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_selection_key: tuple[float, float] | None = None
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics, global_step = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            warmup_base_lrs=warmup_base_lrs,
            warmup_steps=config.warmup_steps,
            global_step=global_step,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_metrics = _evaluate_split(model, val_loader, device)
        val_ood_metrics = _evaluate_split(model, val_ood_loader, device)
        test_iid_metrics = _evaluate_split(model, test_iid_loader, device)
        test_ood_metrics = _evaluate_split(model, test_ood_loader, device)
        selection_key = (
            float(val_ood_metrics["accuracy"]),
            -float(val_ood_metrics["loss"]),
        )
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        history.append(
            RelationEpochRecord(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
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
    elapsed_seconds = time.perf_counter() - start_time
    if best_state_dict is None:
        raise RuntimeError("best_state_dict must be populated after training.")
    model_output_dir = output_dir / name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    selected_checkpoint_path = model_output_dir / "selected_checkpoint.pt"
    torch.save(best_state_dict, selected_checkpoint_path)
    model.load_state_dict(best_state_dict)
    selected_val_ood_metrics = _evaluate_split(model, val_ood_loader, device)
    selected_test_iid_metrics = _evaluate_split(model, test_iid_loader, device)
    selected_test_ood_metrics = _evaluate_split(model, test_ood_loader, device)
    selected_prediction_csv_paths = {
        "val_ood": str(model_output_dir / "selected_val_ood_predictions.csv"),
        "test_iid": str(model_output_dir / "selected_test_iid_predictions.csv"),
        "test_ood": str(model_output_dir / "selected_test_ood_predictions.csv"),
    }
    _export_prediction_csv(
        model=model,
        dataset=val_ood_dataset,
        split_name="val_ood",
        device=device,
        output_path=Path(selected_prediction_csv_paths["val_ood"]),
        batch_size=config.eval_batch_size,
    )
    _export_prediction_csv(
        model=model,
        dataset=test_iid_dataset,
        split_name="test_iid",
        device=device,
        output_path=Path(selected_prediction_csv_paths["test_iid"]),
        batch_size=config.eval_batch_size,
    )
    _export_prediction_csv(
        model=model,
        dataset=test_ood_dataset,
        split_name="test_ood",
        device=device,
        output_path=Path(selected_prediction_csv_paths["test_ood"]),
        batch_size=config.eval_batch_size,
    )
    return RelationModelRun(
        name=name,
        parameter_count=parameter_count,
        history=tuple(history),
        elapsed_seconds=elapsed_seconds,
        selected_epoch=best_epoch,
        selected_val_ood_accuracy=float(selected_val_ood_metrics["accuracy"]),
        selected_test_iid_accuracy=float(selected_test_iid_metrics["accuracy"]),
        selected_test_ood_accuracy=float(selected_test_ood_metrics["accuracy"]),
        selected_checkpoint_path=str(selected_checkpoint_path),
        selected_prediction_csv_paths=selected_prediction_csv_paths,
    )


def run_left_right_relation_benchmark(
    config: LeftRightRelationBenchmarkConfig,
    *,
    datasets_bundle: LeftRightRelationDatasetBundle | None = None,
) -> LeftRightRelationBenchmarkOutputs:
    _validate_config(config)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if datasets_bundle is None:
        datasets_bundle = build_left_right_relation_datasets(config)

    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = _make_data_loader(
        datasets_bundle.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = _make_data_loader(
        datasets_bundle.val_dataset,
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

    transformer_model = TransformerRelationClassifier(config=config).to(device)
    codebook_model = PASCNNRelationCodebookClassifier(config=config).to(device)
    linear_state_model = PASCNNRelationLinearStateClassifier(config=config).to(device)

    transformer_run = _run_model(
        name="transformer_relation_classifier",
        model=transformer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_ood_loader=val_ood_loader,
        test_iid_loader=test_iid_loader,
        test_ood_loader=test_ood_loader,
        val_ood_dataset=datasets_bundle.val_ood_dataset,
        test_iid_dataset=datasets_bundle.test_iid_dataset,
        test_ood_dataset=datasets_bundle.test_ood_dataset,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    codebook_run = _run_model(
        name="pascnn_relation_codebook_classifier",
        model=codebook_model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_ood_loader=val_ood_loader,
        test_iid_loader=test_iid_loader,
        test_ood_loader=test_ood_loader,
        val_ood_dataset=datasets_bundle.val_ood_dataset,
        test_iid_dataset=datasets_bundle.test_iid_dataset,
        test_ood_dataset=datasets_bundle.test_ood_dataset,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    linear_state_run = _run_model(
        name="pascnn_relation_linear_state_classifier",
        model=linear_state_model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_ood_loader=val_ood_loader,
        test_iid_loader=test_iid_loader,
        test_ood_loader=test_ood_loader,
        val_ood_dataset=datasets_bundle.val_ood_dataset,
        test_iid_dataset=datasets_bundle.test_iid_dataset,
        test_ood_dataset=datasets_bundle.test_ood_dataset,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    report_path = output_dir / "left_right_relation_benchmark.json"
    outputs = LeftRightRelationBenchmarkOutputs(
        config=config,
        transformer_run=transformer_run,
        codebook_run=codebook_run,
        linear_state_run=linear_state_run,
        report_path=str(report_path),
    )
    report_path.write_text(
        json.dumps(outputs.to_serializable(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return outputs
