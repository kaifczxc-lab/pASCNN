"""Harder tree structural benchmark: compare root-balance states at matched total depth."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import random

from torch.optim import AdamW

from pascnn.data.tree_dataset import generate_synthetic_tree_exact_depth
from pascnn.data.tree_synth import SyntheticTree
from pascnn.data.tree_synth import TREE_INTERNAL_LABELS
from pascnn.data.tree_synth import serialize_synthetic_tree
from pascnn.training.tree_relation_benchmark import (
    MatchedTransformerTreeRelationClassifier,
    PASCNNTreeRelationCodebookClassifier,
    PASCNNTreeRelationLinearStateClassifier,
    TREE_RELATION_CLASS_CODEBOOK,
    TreeRelationBenchmarkConfig,
    TreeRelationBenchmarkOutputs,
    TreeRelationDataset,
    TreeRelationDatasetBundle,
    TreeRelationModelRun,
    TreeRelationBatch,
    _collate_batches,
    _encode_text_rows,
    _evaluate_split,
    _make_data_loader,
    _resolve_device,
    _run_model,
    _validate_config,
)
from pascnn.data.code_lm import ByteLevelTokenizer
from torch.utils.data import DataLoader


TREE_BALANCE_STATE_NAMES: tuple[str, ...] = (
    "left_heavy",
    "balanced",
    "right_heavy",
)
TREE_BALANCE_RELATION_CLASS_NAMES: tuple[str, ...] = (
    "left_state_less",
    "equal_state",
    "left_state_greater",
)


@dataclass(frozen=True)
class TreeBalanceRelationSampleMetadata:
    index: int
    split_name: str
    target_class: int
    target_name: str
    pair_depth: int
    left_state: int
    left_state_name: str
    right_state: int
    right_state_name: str
    text: str

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


def _relation_class_from_states(left_state: int, right_state: int) -> int:
    if left_state < right_state:
        return 0
    if left_state == right_state:
        return 1
    return 2


def _build_root_state_tree(
    *,
    total_depth: int,
    root_state: int,
    rng: random.Random,
) -> str:
    if total_depth < 1:
        raise ValueError("total_depth must be at least 1.")
    if root_state not in (0, 1, 2):
        raise ValueError("root_state must be one of: 0, 1, 2.")

    max_child_depth = total_depth - 1
    if root_state == 1:
        left_depth = max_child_depth
        right_depth = max_child_depth
    elif root_state == 0:
        left_depth = max_child_depth
        right_depth = rng.randint(0, max_child_depth - 1)
    else:
        left_depth = rng.randint(0, max_child_depth - 1)
        right_depth = max_child_depth

    tree = SyntheticTree(
        label=rng.choice(TREE_INTERNAL_LABELS),
        children=(
            generate_synthetic_tree_exact_depth(left_depth, rng),
            generate_synthetic_tree_exact_depth(right_depth, rng),
        ),
    )
    return serialize_synthetic_tree(tree)


def _format_pair_text(left_text: str, right_text: str) -> str:
    return f"L {left_text}\nR {right_text}"


def _build_split_dataset(
    *,
    split_name: str,
    samples_per_class: int,
    depths: tuple[int, ...],
    split_seed: int,
    sequence_length: int,
    tokenizer: ByteLevelTokenizer,
    seen_pair_texts: set[str],
) -> TreeRelationDataset:
    rng = random.Random(split_seed)
    token_texts: list[str] = []
    targets: list[int] = []
    metadata_rows: list[TreeBalanceRelationSampleMetadata] = []

    for target_class, target_name in enumerate(TREE_BALANCE_RELATION_CLASS_NAMES):
        for _ in range(samples_per_class):
            pair_depth = rng.choice(depths)
            valid_state_pairs = tuple(
                (left_state, right_state)
                for left_state in range(3)
                for right_state in range(3)
                if _relation_class_from_states(left_state, right_state) == target_class
            )
            left_state, right_state = rng.choice(valid_state_pairs)
            left_text = _build_root_state_tree(
                total_depth=pair_depth,
                root_state=left_state,
                rng=rng,
            )
            right_text = _build_root_state_tree(
                total_depth=pair_depth,
                root_state=right_state,
                rng=rng,
            )
            pair_text = _format_pair_text(left_text, right_text)
            attempts = 0
            while pair_text in seen_pair_texts:
                attempts += 1
                if attempts >= 256:
                    raise RuntimeError("could not generate enough unique tree-balance relation pairs.")
                left_text = _build_root_state_tree(
                    total_depth=pair_depth,
                    root_state=left_state,
                    rng=rng,
                )
                right_text = _build_root_state_tree(
                    total_depth=pair_depth,
                    root_state=right_state,
                    rng=rng,
                )
                pair_text = _format_pair_text(left_text, right_text)
            seen_pair_texts.add(pair_text)
            metadata_rows.append(
                TreeBalanceRelationSampleMetadata(
                    index=len(metadata_rows),
                    split_name=split_name,
                    target_class=target_class,
                    target_name=target_name,
                    pair_depth=pair_depth,
                    left_state=left_state,
                    left_state_name=TREE_BALANCE_STATE_NAMES[left_state],
                    right_state=right_state,
                    right_state_name=TREE_BALANCE_STATE_NAMES[right_state],
                    text=pair_text,
                )
            )
            token_texts.append(pair_text)
            targets.append(target_class)

    row_order = list(range(len(metadata_rows)))
    rng.shuffle(row_order)
    shuffled_texts = tuple(token_texts[index] for index in row_order)
    shuffled_targets = [targets[index] for index in row_order]
    shuffled_metadata = tuple(metadata_rows[index] for index in row_order)
    token_ids, attention_mask = _encode_text_rows(
        texts=shuffled_texts,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
    )
    import torch

    return TreeRelationDataset(
        token_ids=token_ids,
        attention_mask=attention_mask,
        target_class=torch.tensor(shuffled_targets, dtype=torch.long),
        metadata=shuffled_metadata,  # type: ignore[arg-type]
    )


def build_tree_balance_relation_datasets(
    config: TreeRelationBenchmarkConfig,
) -> TreeRelationDatasetBundle:
    _validate_config(config)
    tokenizer = ByteLevelTokenizer()
    seen_pair_texts: set[str] = set()
    return TreeRelationDatasetBundle(
        train_dataset=_build_split_dataset(
            split_name="train",
            samples_per_class=config.train_samples_per_class,
            depths=config.train_depths,
            split_seed=config.seed,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            seen_pair_texts=seen_pair_texts,
        ),
        val_iid_dataset=_build_split_dataset(
            split_name="val_iid",
            samples_per_class=config.val_iid_samples_per_class,
            depths=config.train_depths,
            split_seed=config.seed + 1,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            seen_pair_texts=seen_pair_texts,
        ),
        val_ood_dataset=_build_split_dataset(
            split_name="val_ood",
            samples_per_class=config.val_ood_samples_per_class,
            depths=config.ood_depths,
            split_seed=config.seed + 2,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            seen_pair_texts=seen_pair_texts,
        ),
        test_iid_dataset=_build_split_dataset(
            split_name="test_iid",
            samples_per_class=config.test_iid_samples_per_class,
            depths=config.train_depths,
            split_seed=config.seed + 3,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            seen_pair_texts=seen_pair_texts,
        ),
        test_ood_dataset=_build_split_dataset(
            split_name="test_ood",
            samples_per_class=config.test_ood_samples_per_class,
            depths=config.ood_depths,
            split_seed=config.seed + 4,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            seen_pair_texts=seen_pair_texts,
        ),
        vocab_size=tokenizer.vocab_size,
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


def run_tree_balance_relation_benchmark(
    config: TreeRelationBenchmarkConfig,
    datasets_bundle: TreeRelationDatasetBundle | None = None,
) -> TreeRelationBenchmarkOutputs:
    import torch

    _validate_config(config)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "tree_balance_relation_benchmark.json"
    splits_path = output_dir / "tree_balance_relation_splits.json"

    if datasets_bundle is None:
        datasets_bundle = build_tree_balance_relation_datasets(config)
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
        name="matched_transformer_meanpool_tree_balance_relation_classifier",
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
        name="pascnn_tree_balance_relation_codebook_classifier",
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
        name="pascnn_tree_balance_relation_linear_state_classifier",
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
