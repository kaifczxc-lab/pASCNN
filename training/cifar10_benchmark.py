"""Quick CIFAR-10 comparison benchmark for transformer vs pASCNN decoder families."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
import random
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
from torch.utils.data import Subset
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms

from pascnn.core.cell import PASCNNCell
from pascnn.diagnostics import collect_core_diagnostics
from pascnn.types import CellConfig
from pascnn.types import LOGIC_VERTEX_INDICES
from pascnn.types import LogicalReadoutOutputs
from pascnn.types import OpBackend


CIFAR10_NUM_CLASSES = 10
CIFAR10_IMAGE_SIZE = 32
CIFAR10_IMAGE_CHANNELS = 3
TEN_CLASS_TERNARY_CODEBOOK: tuple[tuple[int, ...], ...] = (
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


@dataclass(frozen=True)
class CIFAR10BenchmarkConfig:
    data_root: str
    output_dir: str
    download: bool = False
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    core_learning_rate: float = 1e-3
    core_coherence_learning_rate: float = 3e-3
    core_weight_decay: float = 0.0
    warmup_steps: int = 20
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    seed: int = 7
    val_size: int = 5000
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None
    device: str = "auto"
    patch_size: int = 4
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_hidden_dim: int = 128
    dropout: float = 0.0
    num_logic_heads: int = 5
    core_hidden_dim: int = 16
    core_branch_depth: int = 2
    core_branch_base: int = 2
    core_num_layers: int = 1
    core_backend: OpBackend = "reference"


@dataclass(frozen=True)
class CIFAR10DatasetBundle:
    train_dataset: Dataset[Any]
    val_dataset: Dataset[Any]
    test_dataset: Dataset[Any]


@dataclass(frozen=True)
class CIFAR10ClassifierOutputs:
    class_log_scores: torch.Tensor
    class_probabilities: torch.Tensor
    predicted_class: torch.Tensor
    core_diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class CIFAR10EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float
    edge_prefix_depth_to_uniform_ratio_mean: float | None = None
    wave_prefix_depth_to_uniform_ratio_mean: float | None = None
    edge_defect_norm_sq_normalized_mean: float | None = None
    edge_amplitude_mean: float | None = None

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CIFAR10ModelRun:
    name: str
    parameter_count: int
    history: tuple[CIFAR10EpochRecord, ...]
    elapsed_seconds: float

    def to_serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": self.parameter_count,
            "history": [record.to_serializable() for record in self.history],
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class CIFAR10BenchmarkOutputs:
    config: CIFAR10BenchmarkConfig
    transformer_run: CIFAR10ModelRun
    codebook_run: CIFAR10ModelRun
    linear_state_run: CIFAR10ModelRun
    report_path: str

    def to_serializable(self) -> dict[str, Any]:
        config_dict = asdict(self.config)
        config_dict["data_root"] = str(config_dict["data_root"])
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


def _validate_config(config: CIFAR10BenchmarkConfig) -> None:
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
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if config.val_size < 1:
        raise ValueError("val_size must be at least 1.")
    if config.patch_size < 1 or CIFAR10_IMAGE_SIZE % config.patch_size != 0:
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
    if config.num_logic_heads != len(TEN_CLASS_TERNARY_CODEBOOK[0]):
        raise ValueError(
            f"num_logic_heads must be {len(TEN_CLASS_TERNARY_CODEBOOK[0])} for the fixed 10-class codebook."
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


def _limit_dataset(dataset: Dataset[Any], limit: int | None) -> Dataset[Any]:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise ValueError("dataset limit must be at least 1 when provided.")
    return Subset(dataset, list(range(limit)))


def build_cifar10_datasets(config: CIFAR10BenchmarkConfig) -> CIFAR10DatasetBundle:
    _validate_config(config)
    transform = transforms.ToTensor()
    train_full = datasets.CIFAR10(
        root=config.data_root,
        train=True,
        transform=transform,
        download=config.download,
    )
    if config.val_size >= len(train_full):
        raise ValueError("val_size must be smaller than the CIFAR-10 train split.")
    train_size = len(train_full) - config.val_size
    split_generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(
        train_full,
        [train_size, config.val_size],
        generator=split_generator,
    )
    test_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=False,
        transform=transform,
        download=config.download,
    )
    return CIFAR10DatasetBundle(
        train_dataset=_limit_dataset(train_dataset, config.train_limit),
        val_dataset=_limit_dataset(val_dataset, config.val_limit),
        test_dataset=_limit_dataset(test_dataset, config.test_limit),
    )


def _make_data_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _move_batch_to_device(
    image: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return image.to(device), target.to(device)


class PatchTransformerImageEncoder(nn.Module):
    """Patchify CIFAR-10 images and encode them with a small transformer."""

    def __init__(
        self,
        *,
        patch_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (CIFAR10_IMAGE_SIZE // patch_size) ** 2
        patch_dim = CIFAR10_IMAGE_CHANNELS * patch_size * patch_size
        self.patch_projection = nn.Linear(patch_dim, embedding_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embedding_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embedding_dim)
        nn.init.normal_(self.class_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape [B, 3, 32, 32].")
        if image.shape[1] != CIFAR10_IMAGE_CHANNELS:
            raise ValueError("PatchTransformerImageEncoder expects 3 input channels.")
        if image.shape[2] != CIFAR10_IMAGE_SIZE or image.shape[3] != CIFAR10_IMAGE_SIZE:
            raise ValueError("PatchTransformerImageEncoder expects spatial shape 32x32.")
        if not image.is_floating_point() or image.is_complex():
            raise TypeError("image must be a real floating tensor.")

        patches = F.unfold(
            image,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        token_states = self.patch_projection(patches)
        class_token = self.class_token.expand(image.shape[0], -1, -1)
        token_states = torch.cat((class_token, token_states), dim=1)
        token_states = token_states + self.position_embedding[:, : token_states.shape[1], :]
        token_states = self.encoder(token_states)
        return self.final_norm(token_states[:, 0, :])


class TransformerImageClassifier(nn.Module):
    def __init__(self, *, config: CIFAR10BenchmarkConfig) -> None:
        super().__init__()
        self.encoder = PatchTransformerImageEncoder(
            patch_size=config.patch_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
        )
        self.classifier = nn.Linear(config.embedding_dim, CIFAR10_NUM_CLASSES)

    def forward(self, image: torch.Tensor) -> CIFAR10ClassifierOutputs:
        embedding = self.encoder(image)
        class_log_scores = self.classifier(embedding)
        class_probabilities = torch.softmax(class_log_scores, dim=-1)
        predicted_class = class_log_scores.argmax(dim=-1)
        return CIFAR10ClassifierOutputs(
            class_log_scores=class_log_scores,
            class_probabilities=class_probabilities,
            predicted_class=predicted_class,
        )


class CIFAR10CodebookAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "class_codebook",
            torch.tensor(TEN_CLASS_TERNARY_CODEBOOK, dtype=torch.long),
        )
        self._probability_epsilon = torch.finfo(torch.float32).tiny

    def forward(self, readout: LogicalReadoutOutputs) -> CIFAR10ClassifierOutputs:
        if not isinstance(readout, LogicalReadoutOutputs):
            raise TypeError("readout must be a LogicalReadoutOutputs instance.")
        logic_probabilities = readout.logic_probabilities
        if logic_probabilities.ndim != 3:
            raise ValueError("logic_probabilities must have shape [B, H, 3].")
        if logic_probabilities.shape[1] != len(TEN_CLASS_TERNARY_CODEBOOK[0]):
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
        return CIFAR10ClassifierOutputs(
            class_log_scores=class_log_scores,
            class_probabilities=class_probabilities,
            predicted_class=predicted_class,
        )


class PASCNNCIFARCodebookClassifier(nn.Module):
    def __init__(self, *, config: CIFAR10BenchmarkConfig) -> None:
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
        self.decoder = CIFAR10CodebookAdapter()

    def forward(self, image: torch.Tensor) -> CIFAR10ClassifierOutputs:
        embedding = self.encoder(image)
        cell_outputs = self.core(embedding)
        decode_outputs = self.decoder(cell_outputs.readout)
        diagnostics = collect_core_diagnostics(cell_outputs).as_logging_dict()
        return CIFAR10ClassifierOutputs(
            class_log_scores=decode_outputs.class_log_scores,
            class_probabilities=decode_outputs.class_probabilities,
            predicted_class=decode_outputs.predicted_class,
            core_diagnostics=diagnostics,
        )


class PASCNNCIFARLinearStateClassifier(nn.Module):
    def __init__(self, *, config: CIFAR10BenchmarkConfig) -> None:
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
            CIFAR10_NUM_CLASSES,
        )

    def forward(self, image: torch.Tensor) -> CIFAR10ClassifierOutputs:
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
        return CIFAR10ClassifierOutputs(
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
    config: CIFAR10BenchmarkConfig,
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
    test_loader: DataLoader[Any],
    config: CIFAR10BenchmarkConfig,
    device: torch.device,
) -> CIFAR10ModelRun:
    first_image, _ = next(iter(train_loader))
    first_image = first_image.to(device)
    _ = model(first_image)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer, warmup_base_lrs = _build_optimizer(model, config)
    history: list[CIFAR10EpochRecord] = []
    start_time = time.perf_counter()
    global_step = 0
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
        test_metrics = _evaluate_split(model, test_loader, device)
        history.append(
            CIFAR10EpochRecord(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                test_loss=test_metrics["loss"],
                test_accuracy=test_metrics["accuracy"],
                edge_prefix_depth_to_uniform_ratio_mean=test_metrics.get(
                    "edge_prefix_depth_to_uniform_ratio_mean"
                ),
                wave_prefix_depth_to_uniform_ratio_mean=test_metrics.get(
                    "wave_prefix_depth_to_uniform_ratio_mean"
                ),
                edge_defect_norm_sq_normalized_mean=test_metrics.get(
                    "edge_defect_norm_sq_normalized_mean"
                ),
                edge_amplitude_mean=test_metrics.get("edge_amplitude_mean"),
            )
        )
    elapsed_seconds = time.perf_counter() - start_time
    return CIFAR10ModelRun(
        name=name,
        parameter_count=parameter_count,
        history=tuple(history),
        elapsed_seconds=elapsed_seconds,
    )


def run_cifar10_benchmark(
    config: CIFAR10BenchmarkConfig,
    *,
    datasets_bundle: CIFAR10DatasetBundle | None = None,
) -> CIFAR10BenchmarkOutputs:
    _validate_config(config)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if datasets_bundle is None:
        datasets_bundle = build_cifar10_datasets(config)

    device = _resolve_device(config.device)
    train_loader = _make_data_loader(
        datasets_bundle.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = _make_data_loader(
        datasets_bundle.val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = _make_data_loader(
        datasets_bundle.test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    transformer_model = TransformerImageClassifier(config=config).to(device)
    codebook_model = PASCNNCIFARCodebookClassifier(config=config).to(device)
    linear_state_model = PASCNNCIFARLinearStateClassifier(config=config).to(device)

    transformer_run = _run_model(
        name="transformer_image_classifier",
        model=transformer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )
    codebook_run = _run_model(
        name="pascnn_codebook_classifier",
        model=codebook_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )
    linear_state_run = _run_model(
        name="pascnn_linear_state_classifier",
        model=linear_state_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "cifar10_benchmark.json"
    outputs = CIFAR10BenchmarkOutputs(
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
