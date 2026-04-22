"""Pure PyTorch training and evaluation path for the reference pASCNN MNIST model."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal
from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms

from pascnn.diagnostics import collect_mnist_diagnostics
from pascnn.losses import PASCNNLoss
from pascnn.losses import PASCNNLossConfig
from pascnn.models.mnist import MNISTReferenceModel
from pascnn.types import CellConfig
from pascnn.types import OpBackend


@dataclass(frozen=True)
class MNISTTrainingConfig:
    data_root: str
    output_dir: str
    download: bool = False
    batch_size: int = 64
    eval_batch_size: int = 256
    epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    num_workers: int = 0
    seed: int = 7
    val_size: int = 5000
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None
    device: str = "auto"
    hidden_dim: int = 8
    branch_depth: int = 2
    branch_base: int = 2
    num_layers: int = 1
    num_logic_heads: int = 5
    decoder_mode: Literal["linear_state", "codebook"] = "linear_state"
    core_backend: OpBackend = "reference"
    encoder_embedding_dim: int = 32
    encoder_base_channels: int = 16
    lambda_s: float = 1e-3
    lambda_u: float = 1e-3
    lambda_c: float = 1e-4
    lambda_a: float = 0.0
    resume_from_checkpoint: bool = False


@dataclass(frozen=True)
class MNISTDatasetBundle:
    train_dataset: Dataset[Any]
    val_dataset: Dataset[Any]
    test_dataset: Dataset[Any]


@dataclass(frozen=True)
class MNISTEpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float
    test_confidence_mean: float
    test_one_minus_confidence_mean: float
    test_wave_coherence_gate_mean: float
    test_wave_prefix_depth_mean: float
    test_wave_prefix_depth_normalized_mean: float
    test_wave_prefix_depth_lift_over_uniform_mean: float
    test_wave_prefix_depth_to_uniform_ratio_mean: float
    test_wave_defect_norm_sq_normalized_mean: float
    test_wave_coherence_gate_saturation_rate_95: float
    test_wave_coherence_gate_saturation_rate_99: float
    test_edge_prefix_depth_mean: float
    test_edge_prefix_depth_normalized_mean: float
    test_edge_prefix_depth_to_uniform_ratio_mean: float
    test_edge_defect_norm_sq_normalized_mean: float
    test_edge_coherence_gate_saturation_rate_95: float
    test_edge_coherence_gate_saturation_rate_99: float
    test_code_symbol_balance_mean: float
    test_code_symbol_max_usage_max: float
    test_loss_delta: float
    test_accuracy_delta: float
    best_test_loss_so_far: float
    best_test_accuracy_so_far: float

    def to_serializable(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class MNISTConvergenceDiagnostics:
    epochs_completed: int
    test_loss_history: tuple[float, ...]
    test_accuracy_history: tuple[float, ...]
    test_confidence_history: tuple[float, ...]
    test_one_minus_confidence_history: tuple[float, ...]
    test_wave_coherence_gate_history: tuple[float, ...]
    test_wave_prefix_depth_history: tuple[float, ...]
    test_wave_prefix_depth_normalized_history: tuple[float, ...]
    test_wave_prefix_depth_lift_over_uniform_history: tuple[float, ...]
    test_wave_prefix_depth_to_uniform_ratio_history: tuple[float, ...]
    test_wave_defect_norm_sq_normalized_history: tuple[float, ...]
    test_wave_coherence_gate_saturation_rate_95_history: tuple[float, ...]
    test_wave_coherence_gate_saturation_rate_99_history: tuple[float, ...]
    test_edge_prefix_depth_history: tuple[float, ...]
    test_edge_prefix_depth_normalized_history: tuple[float, ...]
    test_edge_prefix_depth_to_uniform_ratio_history: tuple[float, ...]
    test_edge_defect_norm_sq_normalized_history: tuple[float, ...]
    test_edge_coherence_gate_saturation_rate_95_history: tuple[float, ...]
    test_edge_coherence_gate_saturation_rate_99_history: tuple[float, ...]
    test_code_symbol_balance_history: tuple[float, ...]
    test_code_symbol_max_usage_history: tuple[float, ...]
    initial_test_loss: float
    final_test_loss: float
    best_test_loss: float
    initial_test_accuracy: float
    final_test_accuracy: float
    best_test_accuracy: float
    test_loss_improvement: float
    test_accuracy_gain: float
    best_test_loss_epoch: int
    best_test_accuracy_epoch: int

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MNISTTrainingRunOutputs:
    config: MNISTTrainingConfig
    history: tuple[MNISTEpochRecord, ...]
    convergence: MNISTConvergenceDiagnostics
    output_dir: str
    metrics_path: str
    summary_path: str
    checkpoint_path: str


def _validate_config(config: MNISTTrainingConfig) -> None:
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if config.eval_batch_size < 1:
        raise ValueError("eval_batch_size must be at least 1.")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if config.val_size < 1:
        raise ValueError("val_size must be at least 1.")
    if config.decoder_mode not in ("linear_state", "codebook"):
        raise ValueError("decoder_mode must be 'linear_state' or 'codebook'.")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _limit_dataset(dataset: Dataset[Any], limit: int | None) -> Dataset[Any]:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise ValueError("dataset limit must be at least 1 when provided.")
    return Subset(dataset, list(range(limit)))


def build_mnist_datasets(config: MNISTTrainingConfig) -> MNISTDatasetBundle:
    """Build train/validation/test datasets for an MNIST run."""

    _validate_config(config)
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(
        root=config.data_root,
        train=True,
        transform=transform,
        download=config.download,
    )
    if config.val_size >= len(train_full):
        raise ValueError("val_size must be smaller than the MNIST train split.")

    train_size = len(train_full) - config.val_size
    split_generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(
        train_full,
        [train_size, config.val_size],
        generator=split_generator,
    )
    test_dataset = datasets.MNIST(
        root=config.data_root,
        train=False,
        transform=transform,
        download=config.download,
    )

    return MNISTDatasetBundle(
        train_dataset=_limit_dataset(train_dataset, config.train_limit),
        val_dataset=_limit_dataset(val_dataset, config.val_limit),
        test_dataset=_limit_dataset(test_dataset, config.test_limit),
    )


def _make_data_loader(
    dataset: Dataset[Any],
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


def _build_model_and_loss(
    config: MNISTTrainingConfig,
    device: torch.device,
) -> tuple[MNISTReferenceModel, PASCNNLoss]:
    cell_config = CellConfig(
        hidden_dim=config.hidden_dim,
        branch_depth=config.branch_depth,
        branch_base=config.branch_base,
        num_layers=config.num_layers,
        num_logic_heads=config.num_logic_heads,
        op_backend=config.core_backend,
    )
    model = MNISTReferenceModel(
        cell_config=cell_config,
        encoder_embedding_dim=config.encoder_embedding_dim,
        encoder_base_channels=config.encoder_base_channels,
        decoder_mode=config.decoder_mode,
    ).to(device)
    loss_module = PASCNNLoss(
        PASCNNLossConfig(
            lambda_s=config.lambda_s,
            lambda_u=config.lambda_u,
            lambda_c=config.lambda_c,
            lambda_a=config.lambda_a,
        )
    ).to(device)
    return model, loss_module


def _move_batch_to_device(
    image: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return image.to(device), target.to(device)


def _evaluate_split(
    model: MNISTReferenceModel,
    loss_module: PASCNNLoss,
    data_loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float]:
    if len(data_loader.dataset) == 0:
        raise ValueError("evaluation dataset must not be empty.")

    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    total_confidence = 0.0
    total_one_minus_confidence = 0.0
    total_wave_coherence_gate = 0.0
    total_wave_prefix_depth = 0.0
    total_wave_prefix_depth_normalized = 0.0
    total_wave_prefix_depth_lift_over_uniform = 0.0
    total_wave_prefix_depth_to_uniform_ratio = 0.0
    total_wave_defect_norm_sq_normalized = 0.0
    total_wave_coherence_gate_saturation_rate_95 = 0.0
    total_wave_coherence_gate_saturation_rate_99 = 0.0
    total_edge_prefix_depth = 0.0
    total_edge_prefix_depth_normalized = 0.0
    total_edge_prefix_depth_to_uniform_ratio = 0.0
    total_edge_defect_norm_sq_normalized = 0.0
    total_edge_coherence_gate_saturation_rate_95 = 0.0
    total_edge_coherence_gate_saturation_rate_99 = 0.0
    total_code_symbol_balance = 0.0
    total_code_symbol_max_usage = 0.0

    with torch.inference_mode():
        for image, target in data_loader:
            image, target = _move_batch_to_device(image, target, device)
            outputs = model(image)
            loss_outputs = loss_module(outputs, target)
            diagnostics = collect_mnist_diagnostics(outputs)

            batch_size = target.shape[0]
            total_examples += batch_size
            total_loss += float(loss_outputs.total_loss.detach().cpu()) * batch_size
            total_correct += int((outputs.decode.predicted_class == target).sum().item())
            total_confidence += float(diagnostics.class_confidence.sum().detach().cpu())
            total_one_minus_confidence += float(
                diagnostics.one_minus_confidence.sum().detach().cpu()
            )
            total_wave_coherence_gate += float(
                diagnostics.core.wave_coherence_gate.sum().detach().cpu()
            )
            total_wave_prefix_depth += float(
                diagnostics.core.wave_prefix_depth.sum().detach().cpu()
            )
            total_wave_prefix_depth_normalized += float(
                diagnostics.core.wave_prefix_depth_normalized.sum().detach().cpu()
            )
            total_wave_prefix_depth_lift_over_uniform += float(
                diagnostics.core.wave_prefix_depth_lift_over_uniform.sum().detach().cpu()
            )
            total_wave_prefix_depth_to_uniform_ratio += float(
                diagnostics.core.wave_prefix_depth_to_uniform_ratio.sum().detach().cpu()
            )
            total_wave_defect_norm_sq_normalized += float(
                diagnostics.core.wave_defect_norm_sq_normalized.sum().detach().cpu()
            )
            total_wave_coherence_gate_saturation_rate_95 += float(
                diagnostics.core.wave_coherence_gate_saturation_95.sum().detach().cpu()
            )
            total_wave_coherence_gate_saturation_rate_99 += float(
                diagnostics.core.wave_coherence_gate_saturation_99.sum().detach().cpu()
            )
            total_edge_prefix_depth += float(
                diagnostics.core.edge_prefix_depth.mean(dim=-1).sum().detach().cpu()
            )
            total_edge_prefix_depth_normalized += float(
                diagnostics.core.edge_prefix_depth_normalized.mean(dim=-1)
                .sum()
                .detach()
                .cpu()
            )
            total_edge_prefix_depth_to_uniform_ratio += float(
                diagnostics.core.edge_prefix_depth_to_uniform_ratio.mean(dim=-1)
                .sum()
                .detach()
                .cpu()
            )
            total_edge_defect_norm_sq_normalized += float(
                diagnostics.core.edge_defect_norm_sq_normalized.mean(dim=-1)
                .sum()
                .detach()
                .cpu()
            )
            total_edge_coherence_gate_saturation_rate_95 += float(
                diagnostics.core.edge_coherence_gate_saturation_95.mean(dim=-1)
                .sum()
                .detach()
                .cpu()
            )
            total_edge_coherence_gate_saturation_rate_99 += float(
                diagnostics.core.edge_coherence_gate_saturation_99.mean(dim=-1)
                .sum()
                .detach()
                .cpu()
            )
            total_code_symbol_balance += float(
                diagnostics.core.code_symbol_balance_per_depth.mean().detach().cpu()
            ) * batch_size
            total_code_symbol_max_usage += float(
                diagnostics.core.code_symbol_max_usage_per_depth.max().detach().cpu()
            ) * batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "confidence_mean": total_confidence / total_examples,
        "one_minus_confidence_mean": total_one_minus_confidence / total_examples,
        "wave_coherence_gate_mean": total_wave_coherence_gate / total_examples,
        "wave_prefix_depth_mean": total_wave_prefix_depth / total_examples,
        "wave_prefix_depth_normalized_mean": (
            total_wave_prefix_depth_normalized / total_examples
        ),
        "wave_prefix_depth_lift_over_uniform_mean": (
            total_wave_prefix_depth_lift_over_uniform / total_examples
        ),
        "wave_prefix_depth_to_uniform_ratio_mean": (
            total_wave_prefix_depth_to_uniform_ratio / total_examples
        ),
        "wave_defect_norm_sq_normalized_mean": (
            total_wave_defect_norm_sq_normalized / total_examples
        ),
        "wave_coherence_gate_saturation_rate_95": (
            total_wave_coherence_gate_saturation_rate_95 / total_examples
        ),
        "wave_coherence_gate_saturation_rate_99": (
            total_wave_coherence_gate_saturation_rate_99 / total_examples
        ),
        "edge_prefix_depth_mean": total_edge_prefix_depth / total_examples,
        "edge_prefix_depth_normalized_mean": (
            total_edge_prefix_depth_normalized / total_examples
        ),
        "edge_prefix_depth_to_uniform_ratio_mean": (
            total_edge_prefix_depth_to_uniform_ratio / total_examples
        ),
        "edge_defect_norm_sq_normalized_mean": (
            total_edge_defect_norm_sq_normalized / total_examples
        ),
        "edge_coherence_gate_saturation_rate_95": (
            total_edge_coherence_gate_saturation_rate_95 / total_examples
        ),
        "edge_coherence_gate_saturation_rate_99": (
            total_edge_coherence_gate_saturation_rate_99 / total_examples
        ),
        "code_symbol_balance_mean": total_code_symbol_balance / total_examples,
        "code_symbol_max_usage_max": total_code_symbol_max_usage / total_examples,
    }


def _train_one_epoch(
    model: MNISTReferenceModel,
    loss_module: PASCNNLoss,
    data_loader: DataLoader[Any],
    optimizer: Adam,
    device: torch.device,
) -> dict[str, float]:
    if len(data_loader.dataset) == 0:
        raise ValueError("training dataset must not be empty.")

    model.train()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0

    for image, target in data_loader:
        image, target = _move_batch_to_device(image, target, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        loss_outputs = loss_module(outputs, target)
        loss_outputs.total_loss.backward()
        optimizer.step()

        batch_size = target.shape[0]
        total_examples += batch_size
        total_loss += float(loss_outputs.total_loss.detach().cpu()) * batch_size
        total_correct += int((outputs.decode.predicted_class == target).sum().item())

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def _write_jsonl(path: Path, records: tuple[MNISTEpochRecord, ...]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_serializable(), sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_checkpoint(
    checkpoint_path: Path,
    model: MNISTReferenceModel,
    optimizer: Adam,
    history: tuple[MNISTEpochRecord, ...],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": [record.to_serializable() for record in history],
            "decoder_mode": model.decoder_mode,
        },
        checkpoint_path,
    )


def _load_checkpoint(
    checkpoint_path: Path,
    model: MNISTReferenceModel,
    optimizer: Adam,
    device: torch.device,
) -> tuple[MNISTEpochRecord, ...]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_decoder_mode = checkpoint.get("decoder_mode", "codebook")
    if checkpoint_decoder_mode != model.decoder_mode:
        raise ValueError(
            "Checkpoint decoder_mode does not match model decoder_mode. "
            f"Checkpoint has '{checkpoint_decoder_mode}', model expects '{model.decoder_mode}'."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    normalized_history = []
    for record in checkpoint["history"]:
        if "test_uncertainty_mean" in record:
            record["test_one_minus_confidence_mean"] = record.pop(
                "test_uncertainty_mean"
            )
        if "test_wave_amplitude_mean" in record:
            record["test_wave_coherence_gate_mean"] = record.pop(
                "test_wave_amplitude_mean"
            )
        record.setdefault("test_wave_prefix_depth_mean", 0.0)
        record.setdefault("test_wave_prefix_depth_normalized_mean", 0.0)
        record.setdefault("test_wave_prefix_depth_lift_over_uniform_mean", 0.0)
        record.setdefault("test_wave_prefix_depth_to_uniform_ratio_mean", 0.0)
        record.setdefault("test_wave_defect_norm_sq_normalized_mean", 0.0)
        record.setdefault("test_wave_coherence_gate_saturation_rate_95", 0.0)
        record.setdefault("test_wave_coherence_gate_saturation_rate_99", 0.0)
        record.setdefault("test_edge_prefix_depth_normalized_mean", 0.0)
        record.setdefault("test_edge_prefix_depth_to_uniform_ratio_mean", 0.0)
        record.setdefault("test_edge_defect_norm_sq_normalized_mean", 0.0)
        record.setdefault("test_edge_coherence_gate_saturation_rate_95", 0.0)
        record.setdefault("test_edge_coherence_gate_saturation_rate_99", 0.0)
        record.setdefault("test_code_symbol_balance_mean", 0.0)
        record.setdefault("test_code_symbol_max_usage_max", 0.0)
        normalized_history.append(MNISTEpochRecord(**record))
    return tuple(normalized_history)


def _build_convergence(
    history: tuple[MNISTEpochRecord, ...],
) -> MNISTConvergenceDiagnostics:
    if not history:
        raise ValueError("history must contain at least one epoch record.")

    test_loss_history = tuple(record.test_loss for record in history)
    test_accuracy_history = tuple(record.test_accuracy for record in history)
    test_confidence_history = tuple(record.test_confidence_mean for record in history)
    test_one_minus_confidence_history = tuple(
        record.test_one_minus_confidence_mean for record in history
    )
    test_wave_coherence_gate_history = tuple(
        record.test_wave_coherence_gate_mean for record in history
    )
    test_wave_prefix_depth_history = tuple(
        record.test_wave_prefix_depth_mean for record in history
    )
    test_wave_prefix_depth_normalized_history = tuple(
        record.test_wave_prefix_depth_normalized_mean for record in history
    )
    test_wave_prefix_depth_lift_over_uniform_history = tuple(
        record.test_wave_prefix_depth_lift_over_uniform_mean for record in history
    )
    test_wave_prefix_depth_to_uniform_ratio_history = tuple(
        record.test_wave_prefix_depth_to_uniform_ratio_mean for record in history
    )
    test_wave_defect_norm_sq_normalized_history = tuple(
        record.test_wave_defect_norm_sq_normalized_mean for record in history
    )
    test_wave_coherence_gate_saturation_rate_95_history = tuple(
        record.test_wave_coherence_gate_saturation_rate_95 for record in history
    )
    test_wave_coherence_gate_saturation_rate_99_history = tuple(
        record.test_wave_coherence_gate_saturation_rate_99 for record in history
    )
    test_edge_prefix_depth_history = tuple(
        record.test_edge_prefix_depth_mean for record in history
    )
    test_edge_prefix_depth_normalized_history = tuple(
        record.test_edge_prefix_depth_normalized_mean for record in history
    )
    test_edge_prefix_depth_to_uniform_ratio_history = tuple(
        record.test_edge_prefix_depth_to_uniform_ratio_mean for record in history
    )
    test_edge_defect_norm_sq_normalized_history = tuple(
        record.test_edge_defect_norm_sq_normalized_mean for record in history
    )
    test_edge_coherence_gate_saturation_rate_95_history = tuple(
        record.test_edge_coherence_gate_saturation_rate_95 for record in history
    )
    test_edge_coherence_gate_saturation_rate_99_history = tuple(
        record.test_edge_coherence_gate_saturation_rate_99 for record in history
    )
    test_code_symbol_balance_history = tuple(
        record.test_code_symbol_balance_mean for record in history
    )
    test_code_symbol_max_usage_history = tuple(
        record.test_code_symbol_max_usage_max for record in history
    )

    best_test_loss = min(test_loss_history)
    best_test_accuracy = max(test_accuracy_history)
    best_test_loss_epoch = history[test_loss_history.index(best_test_loss)].epoch
    best_test_accuracy_epoch = history[test_accuracy_history.index(best_test_accuracy)].epoch

    return MNISTConvergenceDiagnostics(
        epochs_completed=len(history),
        test_loss_history=test_loss_history,
        test_accuracy_history=test_accuracy_history,
        test_confidence_history=test_confidence_history,
        test_one_minus_confidence_history=test_one_minus_confidence_history,
        test_wave_coherence_gate_history=test_wave_coherence_gate_history,
        test_wave_prefix_depth_history=test_wave_prefix_depth_history,
        test_wave_prefix_depth_normalized_history=(
            test_wave_prefix_depth_normalized_history
        ),
        test_wave_prefix_depth_lift_over_uniform_history=(
            test_wave_prefix_depth_lift_over_uniform_history
        ),
        test_wave_prefix_depth_to_uniform_ratio_history=(
            test_wave_prefix_depth_to_uniform_ratio_history
        ),
        test_wave_defect_norm_sq_normalized_history=(
            test_wave_defect_norm_sq_normalized_history
        ),
        test_wave_coherence_gate_saturation_rate_95_history=(
            test_wave_coherence_gate_saturation_rate_95_history
        ),
        test_wave_coherence_gate_saturation_rate_99_history=(
            test_wave_coherence_gate_saturation_rate_99_history
        ),
        test_edge_prefix_depth_history=test_edge_prefix_depth_history,
        test_edge_prefix_depth_normalized_history=(
            test_edge_prefix_depth_normalized_history
        ),
        test_edge_prefix_depth_to_uniform_ratio_history=(
            test_edge_prefix_depth_to_uniform_ratio_history
        ),
        test_edge_defect_norm_sq_normalized_history=(
            test_edge_defect_norm_sq_normalized_history
        ),
        test_edge_coherence_gate_saturation_rate_95_history=(
            test_edge_coherence_gate_saturation_rate_95_history
        ),
        test_edge_coherence_gate_saturation_rate_99_history=(
            test_edge_coherence_gate_saturation_rate_99_history
        ),
        test_code_symbol_balance_history=test_code_symbol_balance_history,
        test_code_symbol_max_usage_history=test_code_symbol_max_usage_history,
        initial_test_loss=test_loss_history[0],
        final_test_loss=test_loss_history[-1],
        best_test_loss=best_test_loss,
        initial_test_accuracy=test_accuracy_history[0],
        final_test_accuracy=test_accuracy_history[-1],
        best_test_accuracy=best_test_accuracy,
        test_loss_improvement=test_loss_history[0] - test_loss_history[-1],
        test_accuracy_gain=test_accuracy_history[-1] - test_accuracy_history[0],
        best_test_loss_epoch=best_test_loss_epoch,
        best_test_accuracy_epoch=best_test_accuracy_epoch,
    )


def run_mnist_reference_training(
    config: MNISTTrainingConfig,
    datasets_bundle: MNISTDatasetBundle | None = None,
) -> MNISTTrainingRunOutputs:
    """Run a pure PyTorch MNIST training loop for the reference pASCNN stack."""

    _validate_config(config)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "checkpoint.pt"

    device = _resolve_device(config.device)
    model, loss_module = _build_model_and_loss(config, device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if datasets_bundle is None:
        datasets_bundle = build_mnist_datasets(config)

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

    history: tuple[MNISTEpochRecord, ...] = ()
    start_epoch = 0
    if config.resume_from_checkpoint:
        history = _load_checkpoint(checkpoint_path, model, optimizer, device)
        if history:
            start_epoch = history[-1].epoch

    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_metrics = _train_one_epoch(model, loss_module, train_loader, optimizer, device)
        val_metrics = _evaluate_split(model, loss_module, val_loader, device)
        test_metrics = _evaluate_split(model, loss_module, test_loader, device)

        previous_record = history[-1] if history else None
        test_loss_delta = 0.0
        test_accuracy_delta = 0.0
        if previous_record is not None:
            test_loss_delta = test_metrics["loss"] - previous_record.test_loss
            test_accuracy_delta = test_metrics["accuracy"] - previous_record.test_accuracy

        next_best_test_loss = test_metrics["loss"]
        next_best_test_accuracy = test_metrics["accuracy"]
        if previous_record is not None:
            next_best_test_loss = min(
                previous_record.best_test_loss_so_far,
                next_best_test_loss,
            )
            next_best_test_accuracy = max(
                previous_record.best_test_accuracy_so_far,
                next_best_test_accuracy,
            )

        next_record = MNISTEpochRecord(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_accuracy=train_metrics["accuracy"],
            val_loss=val_metrics["loss"],
            val_accuracy=val_metrics["accuracy"],
            test_loss=test_metrics["loss"],
            test_accuracy=test_metrics["accuracy"],
            test_confidence_mean=test_metrics["confidence_mean"],
            test_one_minus_confidence_mean=test_metrics["one_minus_confidence_mean"],
            test_wave_coherence_gate_mean=test_metrics["wave_coherence_gate_mean"],
            test_wave_prefix_depth_mean=test_metrics["wave_prefix_depth_mean"],
            test_wave_prefix_depth_normalized_mean=test_metrics[
                "wave_prefix_depth_normalized_mean"
            ],
            test_wave_prefix_depth_lift_over_uniform_mean=test_metrics[
                "wave_prefix_depth_lift_over_uniform_mean"
            ],
            test_wave_prefix_depth_to_uniform_ratio_mean=test_metrics[
                "wave_prefix_depth_to_uniform_ratio_mean"
            ],
            test_wave_defect_norm_sq_normalized_mean=test_metrics[
                "wave_defect_norm_sq_normalized_mean"
            ],
            test_wave_coherence_gate_saturation_rate_95=test_metrics[
                "wave_coherence_gate_saturation_rate_95"
            ],
            test_wave_coherence_gate_saturation_rate_99=test_metrics[
                "wave_coherence_gate_saturation_rate_99"
            ],
            test_edge_prefix_depth_mean=test_metrics["edge_prefix_depth_mean"],
            test_edge_prefix_depth_normalized_mean=test_metrics[
                "edge_prefix_depth_normalized_mean"
            ],
            test_edge_prefix_depth_to_uniform_ratio_mean=test_metrics[
                "edge_prefix_depth_to_uniform_ratio_mean"
            ],
            test_edge_defect_norm_sq_normalized_mean=test_metrics[
                "edge_defect_norm_sq_normalized_mean"
            ],
            test_edge_coherence_gate_saturation_rate_95=test_metrics[
                "edge_coherence_gate_saturation_rate_95"
            ],
            test_edge_coherence_gate_saturation_rate_99=test_metrics[
                "edge_coherence_gate_saturation_rate_99"
            ],
            test_code_symbol_balance_mean=test_metrics["code_symbol_balance_mean"],
            test_code_symbol_max_usage_max=test_metrics["code_symbol_max_usage_max"],
            test_loss_delta=test_loss_delta,
            test_accuracy_delta=test_accuracy_delta,
            best_test_loss_so_far=next_best_test_loss,
            best_test_accuracy_so_far=next_best_test_accuracy,
        )
        history = (*history, next_record)
        _write_jsonl(metrics_path, history)
        _save_checkpoint(checkpoint_path, model, optimizer, history)

    convergence = _build_convergence(history)
    _write_json(
        summary_path,
        {
            "config": asdict(config),
            "history": [record.to_serializable() for record in history],
            "convergence": convergence.to_serializable(),
        },
    )
    return MNISTTrainingRunOutputs(
        config=config,
        history=history,
        convergence=convergence,
        output_dir=str(output_dir),
        metrics_path=str(metrics_path),
        summary_path=str(summary_path),
        checkpoint_path=str(checkpoint_path),
    )
