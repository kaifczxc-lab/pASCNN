"""Frozen-cell readout benchmark for isolating decoder bottlenecks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pascnn.adapters.sequence import SequenceBatch
from pascnn.diagnostics import collect_core_diagnostics
from pascnn.training.compositional_ternary_benchmark import (
    CompositionalClassificationBatch,
    CompositionalTernaryBenchmarkConfig,
    PASCNNCompositionalClassifier,
    _make_data_loader,
    _move_batch_to_device,
    build_compositional_ternary_datasets,
)
from pascnn.types import LOGIC_VERTEX_INDICES


@dataclass(frozen=True)
class FrozenCellReadoutBenchmarkConfig:
    base_config: CompositionalTernaryBenchmarkConfig
    head_epochs: int = 50
    head_learning_rate: float = 5e-3
    head_weight_decay: float = 0.0


@dataclass(frozen=True)
class CachedReadoutBatch:
    state_features: torch.Tensor
    born_log_probabilities: torch.Tensor
    target_class: torch.Tensor


class CachedReadoutDataset(Dataset[CachedReadoutBatch]):
    def __init__(
        self,
        *,
        state_features: torch.Tensor,
        born_log_probabilities: torch.Tensor,
        target_class: torch.Tensor,
    ) -> None:
        if state_features.ndim != 2:
            raise ValueError("state_features must have shape [N, F].")
        if born_log_probabilities.ndim != 3:
            raise ValueError("born_log_probabilities must have shape [N, H, 3].")
        if target_class.ndim != 1:
            raise ValueError("target_class must have shape [N].")
        if (
            state_features.shape[0] != born_log_probabilities.shape[0]
            or state_features.shape[0] != target_class.shape[0]
        ):
            raise ValueError("cached tensors must share the same sample axis.")
        self.state_features = state_features
        self.born_log_probabilities = born_log_probabilities
        self.target_class = target_class

    def __len__(self) -> int:
        return int(self.target_class.shape[0])

    def __getitem__(self, index: int) -> CachedReadoutBatch:
        return CachedReadoutBatch(
            state_features=self.state_features[index],
            born_log_probabilities=self.born_log_probabilities[index],
            target_class=self.target_class[index],
        )


@dataclass(frozen=True)
class CachedSplit:
    dataset: CachedReadoutDataset
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class ReadoutRunRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReadoutModelRun:
    name: str
    parameter_count: int
    selected_epoch: int
    history: tuple[ReadoutRunRecord, ...]
    elapsed_seconds: float

    def to_serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": self.parameter_count,
            "selected_epoch": self.selected_epoch,
            "history": [record.to_serializable() for record in self.history],
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class FrozenCellReadoutBenchmarkOutputs:
    config: FrozenCellReadoutBenchmarkConfig
    codebook_size: int
    source_run: ReadoutModelRun
    a3_linear_state_head: ReadoutModelRun
    a1_linear_logp_head: ReadoutModelRun
    a2_soft_codebook_head: ReadoutModelRun
    cached_test_diagnostics: dict[str, float]
    report_path: str

    def to_serializable(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "codebook_size": self.codebook_size,
            "source_run": self.source_run.to_serializable(),
            "a3_linear_state_head": self.a3_linear_state_head.to_serializable(),
            "a1_linear_logp_head": self.a1_linear_logp_head.to_serializable(),
            "a2_soft_codebook_head": self.a2_soft_codebook_head.to_serializable(),
            "cached_test_diagnostics": self.cached_test_diagnostics,
            "report_path": self.report_path,
        }


class LinearStateHead(nn.Module):
    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, batch: CachedReadoutBatch) -> torch.Tensor:
        return self.classifier(batch.state_features)


class LinearLogPHead(nn.Module):
    def __init__(self, *, num_logic_heads: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(num_logic_heads * 3, num_classes)

    def forward(self, batch: CachedReadoutBatch) -> torch.Tensor:
        flattened_logp = batch.born_log_probabilities.reshape(
            batch.born_log_probabilities.shape[0],
            -1,
        )
        return self.classifier(flattened_logp)


class SoftTrainableCodebookHead(nn.Module):
    def __init__(self, *, num_classes: int, num_logic_heads: int) -> None:
        super().__init__()
        self.prototype_logits = nn.Parameter(
            torch.zeros(num_classes, num_logic_heads, 3)
        )
        self.head_weight_logits = nn.Parameter(
            torch.zeros(num_classes, num_logic_heads)
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, batch: CachedReadoutBatch) -> torch.Tensor:
        prototypes = torch.softmax(self.prototype_logits, dim=-1)
        head_weights = torch.sigmoid(self.head_weight_logits)
        logp = batch.born_log_probabilities.unsqueeze(1)
        prototype_scores = (prototypes.unsqueeze(0) * logp).sum(dim=-1)
        weighted_scores = prototype_scores * head_weights.unsqueeze(0)
        return self.bias.unsqueeze(0) + weighted_scores.sum(dim=-1)


def _validate_config(config: FrozenCellReadoutBenchmarkConfig) -> None:
    if config.head_epochs < 1:
        raise ValueError("head_epochs must be at least 1.")
    if config.head_learning_rate <= 0.0:
        raise ValueError("head_learning_rate must be positive.")
    if config.head_weight_decay < 0.0:
        raise ValueError("head_weight_decay must be non-negative.")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _collate_cached_batches(examples: list[CachedReadoutBatch]) -> CachedReadoutBatch:
    return CachedReadoutBatch(
        state_features=torch.stack([example.state_features for example in examples], dim=0),
        born_log_probabilities=torch.stack(
            [example.born_log_probabilities for example in examples],
            dim=0,
        ),
        target_class=torch.stack([example.target_class for example in examples], dim=0),
    )


def _make_cached_loader(
    dataset: CachedReadoutDataset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[CachedReadoutBatch]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate_cached_batches,
        drop_last=False,
        persistent_workers=False,
    )


def _move_cached_batch_to_device(
    batch: CachedReadoutBatch,
    device: torch.device,
) -> CachedReadoutBatch:
    return CachedReadoutBatch(
        state_features=batch.state_features.to(device),
        born_log_probabilities=batch.born_log_probabilities.to(device),
        target_class=batch.target_class.to(device),
    )


def _evaluate_source_model(
    model: PASCNNCompositionalClassifier,
    data_loader: DataLoader[CompositionalClassificationBatch],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.inference_mode():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch)
            loss = F.cross_entropy(outputs.class_log_scores, batch.target_class)
            batch_size = batch.target_class.shape[0]
            total_examples += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((outputs.predicted_class == batch.target_class).sum().item())
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def _train_source_epoch(
    model: PASCNNCompositionalClassifier,
    data_loader: DataLoader[CompositionalClassificationBatch],
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


def _train_source_model(
    model: PASCNNCompositionalClassifier,
    train_loader: DataLoader[CompositionalClassificationBatch],
    val_loader: DataLoader[CompositionalClassificationBatch],
    test_loader: DataLoader[CompositionalClassificationBatch],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> ReadoutModelRun:
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[ReadoutRunRecord] = []
    best_val_loss = float("inf")
    best_epoch = 1
    best_state = deepcopy(model.state_dict())
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _train_source_epoch(model, train_loader, optimizer, device)
        val_metrics = _evaluate_source_model(model, val_loader, device)
        test_metrics = _evaluate_source_model(model, test_loader, device)
        history.append(
            ReadoutRunRecord(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                test_loss=test_metrics["loss"],
                test_accuracy=test_metrics["accuracy"],
            )
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return ReadoutModelRun(
        name="a0_source_pascnn_codebook",
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        selected_epoch=best_epoch,
        history=tuple(history),
        elapsed_seconds=time.perf_counter() - start_time,
    )


def _cache_split(
    source_model: PASCNNCompositionalClassifier,
    data_loader: DataLoader[CompositionalClassificationBatch],
    device: torch.device,
) -> CachedSplit:
    state_feature_rows: list[torch.Tensor] = []
    born_log_probability_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    diagnostics_sums: dict[str, float] = {}
    diagnostics_batches = 0
    probability_epsilon = torch.finfo(torch.float32).tiny
    source_model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            sample_outputs = source_model.model(
                SequenceBatch(
                    token_ids=batch.token_ids,
                    attention_mask=batch.attention_mask,
                )
            )
            logical_vertex_state = sample_outputs.cell.vertex_state[:, LOGIC_VERTEX_INDICES, :]
            state_features = torch.cat(
                (logical_vertex_state.real, logical_vertex_state.imag),
                dim=-1,
            ).reshape(logical_vertex_state.shape[0], -1)
            born_log_probabilities = sample_outputs.cell.readout.logic_probabilities.clamp_min(
                probability_epsilon
            ).log()
            state_feature_rows.append(state_features.detach().cpu())
            born_log_probability_rows.append(born_log_probabilities.detach().cpu())
            target_rows.append(batch.target_class.detach().cpu())
            diagnostics = collect_core_diagnostics(sample_outputs.cell).as_logging_dict()
            diagnostics_batches += 1
            for key, value in diagnostics.items():
                diagnostics_sums[key] = diagnostics_sums.get(key, 0.0) + float(value)

    dataset = CachedReadoutDataset(
        state_features=torch.cat(state_feature_rows, dim=0),
        born_log_probabilities=torch.cat(born_log_probability_rows, dim=0),
        target_class=torch.cat(target_rows, dim=0),
    )
    return CachedSplit(
        dataset=dataset,
        diagnostics={
            key: value / diagnostics_batches
            for key, value in diagnostics_sums.items()
        },
    )


def _evaluate_cached_model(
    model: nn.Module,
    data_loader: DataLoader[CachedReadoutBatch],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.inference_mode():
        for batch in data_loader:
            batch = _move_cached_batch_to_device(batch, device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.target_class)
            batch_size = batch.target_class.shape[0]
            total_examples += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((logits.argmax(dim=-1) == batch.target_class).sum().item())
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def _train_cached_epoch(
    model: nn.Module,
    data_loader: DataLoader[CachedReadoutBatch],
    optimizer: AdamW,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in data_loader:
        batch = _move_cached_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.target_class)
        loss.backward()
        optimizer.step()
        batch_size = batch.target_class.shape[0]
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=-1) == batch.target_class).sum().item())
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def _train_cached_head(
    *,
    name: str,
    model: nn.Module,
    train_loader: DataLoader[CachedReadoutBatch],
    val_loader: DataLoader[CachedReadoutBatch],
    test_loader: DataLoader[CachedReadoutBatch],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> ReadoutModelRun:
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[ReadoutRunRecord] = []
    best_val_loss = float("inf")
    best_epoch = 1
    best_state = deepcopy(model.state_dict())
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _train_cached_epoch(model, train_loader, optimizer, device)
        val_metrics = _evaluate_cached_model(model, val_loader, device)
        test_metrics = _evaluate_cached_model(model, test_loader, device)
        history.append(
            ReadoutRunRecord(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                test_loss=test_metrics["loss"],
                test_accuracy=test_metrics["accuracy"],
            )
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return ReadoutModelRun(
        name=name,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        selected_epoch=best_epoch,
        history=tuple(history),
        elapsed_seconds=time.perf_counter() - start_time,
    )


def run_frozen_cell_readout_benchmark(
    config: FrozenCellReadoutBenchmarkConfig,
) -> FrozenCellReadoutBenchmarkOutputs:
    _validate_config(config)
    base_config = config.base_config
    torch.manual_seed(base_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_config.seed)

    output_dir = Path(base_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "frozen_cell_readout_benchmark.json"

    datasets_bundle = build_compositional_ternary_datasets(base_config)
    device = _resolve_device(base_config.device)
    train_loader = _make_data_loader(
        datasets_bundle.train_dataset,
        batch_size=base_config.batch_size,
        shuffle=True,
    )
    val_loader = _make_data_loader(
        datasets_bundle.val_dataset,
        batch_size=base_config.batch_size,
        shuffle=False,
    )
    test_loader = _make_data_loader(
        datasets_bundle.test_dataset,
        batch_size=base_config.batch_size,
        shuffle=False,
    )

    source_model = PASCNNCompositionalClassifier(
        vocab_size=datasets_bundle.vocab_size,
        codebook=datasets_bundle.codebook,
        config=base_config,
    ).to(device)
    source_run = _train_source_model(
        source_model,
        train_loader,
        val_loader,
        test_loader,
        device,
        epochs=base_config.epochs,
        learning_rate=base_config.learning_rate,
        weight_decay=base_config.weight_decay,
    )

    cached_train = _cache_split(source_model, train_loader, device)
    cached_val = _cache_split(source_model, val_loader, device)
    cached_test = _cache_split(source_model, test_loader, device)
    cached_train_loader = _make_cached_loader(
        cached_train.dataset,
        batch_size=base_config.batch_size,
        shuffle=True,
    )
    cached_val_loader = _make_cached_loader(
        cached_val.dataset,
        batch_size=base_config.batch_size,
        shuffle=False,
    )
    cached_test_loader = _make_cached_loader(
        cached_test.dataset,
        batch_size=base_config.batch_size,
        shuffle=False,
    )

    num_classes = len(datasets_bundle.codebook)
    num_logic_heads = base_config.num_logic_heads
    state_input_dim = cached_train.dataset.state_features.shape[-1]

    a3_run = _train_cached_head(
        name="a3_linear_state_head",
        model=LinearStateHead(input_dim=state_input_dim, num_classes=num_classes).to(device),
        train_loader=cached_train_loader,
        val_loader=cached_val_loader,
        test_loader=cached_test_loader,
        device=device,
        epochs=config.head_epochs,
        learning_rate=config.head_learning_rate,
        weight_decay=config.head_weight_decay,
    )
    a1_run = _train_cached_head(
        name="a1_linear_logp_head",
        model=LinearLogPHead(num_logic_heads=num_logic_heads, num_classes=num_classes).to(
            device
        ),
        train_loader=cached_train_loader,
        val_loader=cached_val_loader,
        test_loader=cached_test_loader,
        device=device,
        epochs=config.head_epochs,
        learning_rate=config.head_learning_rate,
        weight_decay=config.head_weight_decay,
    )
    a2_run = _train_cached_head(
        name="a2_soft_trainable_codebook_head",
        model=SoftTrainableCodebookHead(
            num_classes=num_classes,
            num_logic_heads=num_logic_heads,
        ).to(device),
        train_loader=cached_train_loader,
        val_loader=cached_val_loader,
        test_loader=cached_test_loader,
        device=device,
        epochs=config.head_epochs,
        learning_rate=config.head_learning_rate,
        weight_decay=config.head_weight_decay,
    )

    outputs = FrozenCellReadoutBenchmarkOutputs(
        config=config,
        codebook_size=num_classes,
        source_run=source_run,
        a3_linear_state_head=a3_run,
        a1_linear_logp_head=a1_run,
        a2_soft_codebook_head=a2_run,
        cached_test_diagnostics=cached_test.diagnostics,
        report_path=str(report_path),
    )
    report_path.write_text(
        json.dumps(outputs.to_serializable(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return outputs
