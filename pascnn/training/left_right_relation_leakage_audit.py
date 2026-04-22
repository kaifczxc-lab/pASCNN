"""Leakage audit for the synthetic left-right relation benchmark."""

from __future__ import annotations

import csv
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pascnn.training.left_right_relation_benchmark import (
    LeftRightRelationBenchmarkConfig,
    LeftRightRelationDataset,
    PASCNNRelationCodebookClassifier,
    PASCNNRelationLinearStateClassifier,
    RELATION_IMAGE_SIZE,
    RELATION_NUM_CLASSES,
    TransformerRelationClassifier,
    _resolve_device,
    build_left_right_relation_datasets,
)


@dataclass(frozen=True)
class LeakageAuditModelResult:
    name: str
    json_selected_val_ood_accuracy: float
    json_selected_test_iid_accuracy: float
    json_selected_test_ood_accuracy: float
    csv_selected_val_ood_accuracy: float
    csv_selected_test_iid_accuracy: float
    csv_selected_test_ood_accuracy: float
    checkpoint_selected_val_ood_accuracy: float
    checkpoint_selected_test_iid_accuracy: float
    checkpoint_selected_test_ood_accuracy: float
    left_only_test_ood_accuracy: float
    right_only_test_ood_accuracy: float
    broken_pair_test_ood_accuracy: float

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeakageAuditOutputs:
    benchmark_report_path: str
    chance_accuracy: float
    duplicate_hash_counts: dict[str, int]
    transformer_result: LeakageAuditModelResult
    codebook_result: LeakageAuditModelResult
    linear_state_result: LeakageAuditModelResult
    report_path: str

    def to_serializable(self) -> dict[str, Any]:
        return {
            "benchmark_report_path": self.benchmark_report_path,
            "chance_accuracy": self.chance_accuracy,
            "duplicate_hash_counts": dict(self.duplicate_hash_counts),
            "transformer_result": self.transformer_result.to_serializable(),
            "codebook_result": self.codebook_result.to_serializable(),
            "linear_state_result": self.linear_state_result.to_serializable(),
            "report_path": self.report_path,
        }


class _HalfMaskedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        base_dataset: LeftRightRelationDataset,
        *,
        zero_left: bool,
        zero_right: bool,
    ) -> None:
        self.base_dataset = base_dataset
        self.zero_left = zero_left
        self.zero_right = zero_right

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = self.base_dataset[index]
        image = image.clone()
        midpoint = RELATION_IMAGE_SIZE // 2
        if self.zero_left:
            image[:, :, :midpoint] = 0.0
        if self.zero_right:
            image[:, :, midpoint:] = 0.0
        return image, target


class _BrokenPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, base_dataset: LeftRightRelationDataset) -> None:
        if len(base_dataset) < 2:
            raise ValueError("Broken-pair audit requires at least two samples.")
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = self.base_dataset[index]
        paired_image, _ = self.base_dataset[(index + 1) % len(self.base_dataset)]
        image = image.clone()
        midpoint = RELATION_IMAGE_SIZE // 2
        image[:, :, midpoint:] = paired_image[:, :, midpoint:]
        return image, target


def _load_report(report_path: Path) -> dict[str, Any]:
    return json.loads(report_path.read_text(encoding="utf-8"))


def _config_from_report(report_payload: dict[str, Any]) -> LeftRightRelationBenchmarkConfig:
    return LeftRightRelationBenchmarkConfig(**report_payload["config"])


def _build_model(model_name: str, config: LeftRightRelationBenchmarkConfig) -> nn.Module:
    if model_name == "transformer_relation_classifier":
        return TransformerRelationClassifier(config=config)
    if model_name == "pascnn_relation_codebook_classifier":
        return PASCNNRelationCodebookClassifier(config=config)
    if model_name == "pascnn_relation_linear_state_classifier":
        return PASCNNRelationLinearStateClassifier(config=config)
    raise ValueError(f"Unsupported model name: {model_name}")


def _evaluate_accuracy(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    model.eval()
    total_examples = 0
    total_correct = 0
    with torch.inference_mode():
        for image, target in data_loader:
            image = image.to(device)
            target = target.to(device)
            outputs = model(image)
            total_examples += target.shape[0]
            total_correct += int((outputs.predicted_class == target).sum().item())
    return total_correct / total_examples


def _recompute_csv_accuracy(csv_path: Path) -> float:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {csv_path}")
    correct_sum = sum(int(row["correct"]) for row in rows)
    return correct_sum / len(rows)


def _dataset_hashes(dataset: LeftRightRelationDataset) -> set[str]:
    hashes: set[str] = set()
    for index in range(len(dataset)):
        image, _ = dataset[index]
        image_bytes = image.contiguous().numpy().tobytes()
        hashes.add(hashlib.sha1(image_bytes).hexdigest())
    return hashes


def _duplicate_hash_counts(
    *,
    train_dataset: LeftRightRelationDataset,
    val_dataset: LeftRightRelationDataset,
    val_ood_dataset: LeftRightRelationDataset,
    test_iid_dataset: LeftRightRelationDataset,
    test_ood_dataset: LeftRightRelationDataset,
) -> dict[str, int]:
    train_hashes = _dataset_hashes(train_dataset)
    val_hashes = _dataset_hashes(val_dataset)
    val_ood_hashes = _dataset_hashes(val_ood_dataset)
    test_iid_hashes = _dataset_hashes(test_iid_dataset)
    test_ood_hashes = _dataset_hashes(test_ood_dataset)
    return {
        "train_val": len(train_hashes.intersection(val_hashes)),
        "train_val_ood": len(train_hashes.intersection(val_ood_hashes)),
        "train_test_iid": len(train_hashes.intersection(test_iid_hashes)),
        "train_test_ood": len(train_hashes.intersection(test_ood_hashes)),
        "val_ood_test_ood": len(val_ood_hashes.intersection(test_ood_hashes)),
        "test_iid_test_ood": len(test_iid_hashes.intersection(test_ood_hashes)),
    }


def _run_model_audit(
    *,
    model_payload: dict[str, Any],
    config: LeftRightRelationBenchmarkConfig,
    val_ood_dataset: LeftRightRelationDataset,
    test_iid_dataset: LeftRightRelationDataset,
    test_ood_dataset: LeftRightRelationDataset,
    device: torch.device,
) -> LeakageAuditModelResult:
    model_name = str(model_payload["name"])
    model = _build_model(model_name, config).to(device)
    checkpoint_path = Path(model_payload["selected_checkpoint_path"])
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    val_ood_accuracy = _evaluate_accuracy(
        model,
        val_ood_dataset,
        device=device,
        batch_size=config.eval_batch_size,
    )
    test_iid_accuracy = _evaluate_accuracy(
        model,
        test_iid_dataset,
        device=device,
        batch_size=config.eval_batch_size,
    )
    test_ood_accuracy = _evaluate_accuracy(
        model,
        test_ood_dataset,
        device=device,
        batch_size=config.eval_batch_size,
    )
    left_only_accuracy = _evaluate_accuracy(
        model,
        _HalfMaskedDataset(test_ood_dataset, zero_left=False, zero_right=True),
        device=device,
        batch_size=config.eval_batch_size,
    )
    right_only_accuracy = _evaluate_accuracy(
        model,
        _HalfMaskedDataset(test_ood_dataset, zero_left=True, zero_right=False),
        device=device,
        batch_size=config.eval_batch_size,
    )
    broken_pair_accuracy = _evaluate_accuracy(
        model,
        _BrokenPairDataset(test_ood_dataset),
        device=device,
        batch_size=config.eval_batch_size,
    )
    csv_paths = model_payload["selected_prediction_csv_paths"]
    return LeakageAuditModelResult(
        name=model_name,
        json_selected_val_ood_accuracy=float(model_payload["selected_val_ood_accuracy"]),
        json_selected_test_iid_accuracy=float(model_payload["selected_test_iid_accuracy"]),
        json_selected_test_ood_accuracy=float(model_payload["selected_test_ood_accuracy"]),
        csv_selected_val_ood_accuracy=_recompute_csv_accuracy(Path(csv_paths["val_ood"])),
        csv_selected_test_iid_accuracy=_recompute_csv_accuracy(Path(csv_paths["test_iid"])),
        csv_selected_test_ood_accuracy=_recompute_csv_accuracy(Path(csv_paths["test_ood"])),
        checkpoint_selected_val_ood_accuracy=val_ood_accuracy,
        checkpoint_selected_test_iid_accuracy=test_iid_accuracy,
        checkpoint_selected_test_ood_accuracy=test_ood_accuracy,
        left_only_test_ood_accuracy=left_only_accuracy,
        right_only_test_ood_accuracy=right_only_accuracy,
        broken_pair_test_ood_accuracy=broken_pair_accuracy,
    )


def run_left_right_relation_leakage_audit(
    *,
    benchmark_report_path: str,
    output_path: str | None = None,
) -> LeakageAuditOutputs:
    report_path = Path(benchmark_report_path)
    report_payload = _load_report(report_path)
    config = _config_from_report(report_payload)
    device = _resolve_device(config.device)
    datasets_bundle = build_left_right_relation_datasets(config)
    duplicate_hash_counts = _duplicate_hash_counts(
        train_dataset=datasets_bundle.train_dataset,
        val_dataset=datasets_bundle.val_dataset,
        val_ood_dataset=datasets_bundle.val_ood_dataset,
        test_iid_dataset=datasets_bundle.test_iid_dataset,
        test_ood_dataset=datasets_bundle.test_ood_dataset,
    )
    transformer_result = _run_model_audit(
        model_payload=report_payload["transformer_run"],
        config=config,
        val_ood_dataset=datasets_bundle.val_ood_dataset,
        test_iid_dataset=datasets_bundle.test_iid_dataset,
        test_ood_dataset=datasets_bundle.test_ood_dataset,
        device=device,
    )
    codebook_result = _run_model_audit(
        model_payload=report_payload["codebook_run"],
        config=config,
        val_ood_dataset=datasets_bundle.val_ood_dataset,
        test_iid_dataset=datasets_bundle.test_iid_dataset,
        test_ood_dataset=datasets_bundle.test_ood_dataset,
        device=device,
    )
    linear_state_result = _run_model_audit(
        model_payload=report_payload["linear_state_run"],
        config=config,
        val_ood_dataset=datasets_bundle.val_ood_dataset,
        test_iid_dataset=datasets_bundle.test_iid_dataset,
        test_ood_dataset=datasets_bundle.test_ood_dataset,
        device=device,
    )
    if output_path is None:
        output_path = str(report_path.parent / "left_right_relation_leakage_audit.json")
    outputs = LeakageAuditOutputs(
        benchmark_report_path=str(report_path),
        chance_accuracy=1.0 / float(RELATION_NUM_CLASSES),
        duplicate_hash_counts=duplicate_hash_counts,
        transformer_result=transformer_result,
        codebook_result=codebook_result,
        linear_state_result=linear_state_result,
        report_path=str(output_path),
    )
    Path(output_path).write_text(
        json.dumps(outputs.to_serializable(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return outputs
