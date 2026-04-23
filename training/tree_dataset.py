"""Synthetic tree dataset generator with depth-controlled splits."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import random

from pascnn.data.tree_synth import SyntheticTree
from pascnn.data.tree_synth import TREE_INTERNAL_LABELS
from pascnn.data.tree_synth import TREE_LEAF_LABELS
from pascnn.data.tree_synth import serialize_synthetic_tree
from pascnn.data.tree_synth import synthetic_tree_depth
from pascnn.data.tree_synth import synthetic_tree_leaf_count
from pascnn.data.tree_synth import synthetic_tree_node_count


@dataclass(frozen=True)
class SyntheticTreeTextSample:
    text: str
    depth: int
    node_count: int
    leaf_count: int

    def to_serializable(self) -> dict[str, int | str]:
        return {
            "text": self.text,
            "depth": self.depth,
            "node_count": self.node_count,
            "leaf_count": self.leaf_count,
        }


@dataclass(frozen=True)
class SyntheticTreeDepthBucket:
    depth: int
    samples: tuple[SyntheticTreeTextSample, ...]

    def to_serializable(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "samples": [sample.to_serializable() for sample in self.samples],
        }


@dataclass(frozen=True)
class SyntheticTreeBenchmarkConfig:
    train_max_depth: int
    train_samples_per_depth: int
    validation_samples_per_depth: int
    extrapolation_depths: tuple[int, ...]
    extrapolation_samples_per_depth: int
    train_depths: tuple[int, ...] | None = None
    validation_depths: tuple[int, ...] | None = None
    seed: int = 7


@dataclass(frozen=True)
class SyntheticTreeBenchmarkSplits:
    train_samples: tuple[SyntheticTreeTextSample, ...]
    validation_samples: tuple[SyntheticTreeTextSample, ...]
    extrapolation_buckets: tuple[SyntheticTreeDepthBucket, ...]

    def to_serializable(self) -> dict[str, object]:
        return {
            "train_samples": [sample.to_serializable() for sample in self.train_samples],
            "validation_samples": [
                sample.to_serializable()
                for sample in self.validation_samples
            ],
            "extrapolation_buckets": [
                bucket.to_serializable()
                for bucket in self.extrapolation_buckets
            ],
        }


@dataclass(frozen=True)
class SyntheticTreeBenchmarkArtifacts:
    output_dir: str
    config_path: str
    manifest_path: str
    train_path: str
    validation_path: str
    extrapolation_paths: tuple[tuple[int, str], ...]

    def to_serializable(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "config_path": self.config_path,
            "manifest_path": self.manifest_path,
            "train_path": self.train_path,
            "validation_path": self.validation_path,
            "extrapolation_paths": [
                {
                    "depth": depth,
                    "path": path,
                }
                for depth, path in self.extrapolation_paths
            ],
        }


def _count_exact_depth_trees(depth: int) -> int:
    if depth < 0:
        return 0

    counts: list[int] = [len(TREE_LEAF_LABELS)]
    cumulative_counts: list[int] = [counts[0]]
    for current_depth in range(1, depth + 1):
        previous_cumulative = cumulative_counts[current_depth - 1]
        before_previous_cumulative = (
            cumulative_counts[current_depth - 2]
            if current_depth >= 2
            else 0
        )
        exact_count = len(TREE_INTERNAL_LABELS) * (
            previous_cumulative * previous_cumulative
            - before_previous_cumulative * before_previous_cumulative
        )
        counts.append(exact_count)
        cumulative_counts.append(previous_cumulative + exact_count)
    return counts[depth]


def _validate_depth_sample_capacity(
    *,
    depth: int,
    requested_samples: int,
    split_name: str,
) -> None:
    capacity = _count_exact_depth_trees(depth)
    if requested_samples > capacity:
        raise ValueError(
            f"{split_name} requests {requested_samples} samples at depth {depth}, "
            f"but only {capacity} unique trees exist."
        )


def _validate_benchmark_config(config: SyntheticTreeBenchmarkConfig) -> None:
    if config.train_max_depth < 0:
        raise ValueError("train_max_depth must be non-negative.")
    if config.train_samples_per_depth < 1:
        raise ValueError("train_samples_per_depth must be at least 1.")
    if config.validation_samples_per_depth < 1:
        raise ValueError("validation_samples_per_depth must be at least 1.")
    if config.extrapolation_samples_per_depth < 1:
        raise ValueError("extrapolation_samples_per_depth must be at least 1.")
    train_depths = _resolve_split_depths(
        explicit_depths=config.train_depths,
        fallback_max_depth=config.train_max_depth,
        split_name="train",
    )
    validation_depths = _resolve_split_depths(
        explicit_depths=config.validation_depths,
        fallback_max_depth=config.train_max_depth,
        split_name="validation",
        fallback_depths=train_depths,
    )
    if not config.extrapolation_depths:
        raise ValueError("extrapolation_depths must contain at least one depth.")
    if any(depth < 0 for depth in config.extrapolation_depths):
        raise ValueError("extrapolation depths must be non-negative.")
    if len(set(config.extrapolation_depths)) != len(config.extrapolation_depths):
        raise ValueError("extrapolation_depths must be unique.")
    observed_depths = set(train_depths).union(validation_depths)
    max_observed_depth = max(observed_depths)
    if any(depth <= max_observed_depth for depth in config.extrapolation_depths):
        raise ValueError(
            "every extrapolation depth must be greater than every train/validation depth."
        )
    for depth in observed_depths:
        requested_samples = 0
        if depth in train_depths:
            requested_samples += config.train_samples_per_depth
        if depth in validation_depths:
            requested_samples += config.validation_samples_per_depth
        _validate_depth_sample_capacity(
            depth=depth,
            requested_samples=requested_samples,
            split_name="train/validation",
        )
    for depth in config.extrapolation_depths:
        _validate_depth_sample_capacity(
            depth=depth,
            requested_samples=config.extrapolation_samples_per_depth,
            split_name="extrapolation",
        )


def _resolve_split_depths(
    *,
    explicit_depths: tuple[int, ...] | None,
    fallback_max_depth: int,
    split_name: str,
    fallback_depths: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    if explicit_depths is None:
        if fallback_depths is not None:
            return fallback_depths
        return tuple(range(fallback_max_depth + 1))
    if not explicit_depths:
        raise ValueError(f"{split_name}_depths must contain at least one depth when provided.")
    if any(depth < 0 for depth in explicit_depths):
        raise ValueError(f"{split_name}_depths must be non-negative.")
    if len(set(explicit_depths)) != len(explicit_depths):
        raise ValueError(f"{split_name}_depths must be unique.")
    return tuple(explicit_depths)


def generate_synthetic_tree_exact_depth(
    depth: int,
    rng: random.Random,
) -> SyntheticTree:
    if depth < 0:
        raise ValueError("depth must be non-negative.")
    if depth == 0:
        return SyntheticTree(label=rng.choice(TREE_LEAF_LABELS))

    left_depth = rng.randint(0, depth - 1)
    right_depth = depth - 1
    if left_depth == depth - 1:
        right_depth = rng.randint(0, depth - 1)
    if rng.random() < 0.5:
        left_depth, right_depth = right_depth, left_depth

    return SyntheticTree(
        label=rng.choice(TREE_INTERNAL_LABELS),
        children=(
            generate_synthetic_tree_exact_depth(left_depth, rng),
            generate_synthetic_tree_exact_depth(right_depth, rng),
        ),
    )


def _tree_to_sample(tree: SyntheticTree) -> SyntheticTreeTextSample:
    return SyntheticTreeTextSample(
        text=serialize_synthetic_tree(tree),
        depth=synthetic_tree_depth(tree),
        node_count=synthetic_tree_node_count(tree),
        leaf_count=synthetic_tree_leaf_count(tree),
    )


def generate_unique_synthetic_tree_samples(
    *,
    depth: int,
    num_samples: int,
    rng: random.Random,
    seen_texts: set[str] | None = None,
) -> tuple[SyntheticTreeTextSample, ...]:
    if depth < 0:
        raise ValueError("depth must be non-negative.")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    local_seen_texts = set() if seen_texts is None else seen_texts
    samples: list[SyntheticTreeTextSample] = []
    max_attempts = max(100, num_samples * 200)
    attempts = 0
    while len(samples) < num_samples:
        if attempts >= max_attempts:
            raise RuntimeError("could not generate enough unique tree samples.")
        attempts += 1
        sample = _tree_to_sample(generate_synthetic_tree_exact_depth(depth, rng))
        if sample.depth != depth:
            raise RuntimeError("generated tree depth does not match requested depth.")
        if sample.text in local_seen_texts:
            continue
        local_seen_texts.add(sample.text)
        samples.append(sample)
    return tuple(samples)


def build_synthetic_tree_benchmark(
    config: SyntheticTreeBenchmarkConfig,
) -> SyntheticTreeBenchmarkSplits:
    _validate_benchmark_config(config)
    rng = random.Random(config.seed)
    seen_texts: set[str] = set()
    train_depths = _resolve_split_depths(
        explicit_depths=config.train_depths,
        fallback_max_depth=config.train_max_depth,
        split_name="train",
    )
    validation_depths = _resolve_split_depths(
        explicit_depths=config.validation_depths,
        fallback_max_depth=config.train_max_depth,
        split_name="validation",
        fallback_depths=train_depths,
    )

    train_samples: list[SyntheticTreeTextSample] = []
    validation_samples: list[SyntheticTreeTextSample] = []
    for depth in train_depths:
        train_samples.extend(
            generate_unique_synthetic_tree_samples(
                depth=depth,
                num_samples=config.train_samples_per_depth,
                rng=rng,
                seen_texts=seen_texts,
            )
        )
    for depth in validation_depths:
        validation_samples.extend(
            generate_unique_synthetic_tree_samples(
                depth=depth,
                num_samples=config.validation_samples_per_depth,
                rng=rng,
                seen_texts=seen_texts,
            )
        )

    extrapolation_buckets: list[SyntheticTreeDepthBucket] = []
    for depth in config.extrapolation_depths:
        samples = generate_unique_synthetic_tree_samples(
            depth=depth,
            num_samples=config.extrapolation_samples_per_depth,
            rng=rng,
            seen_texts=seen_texts,
        )
        extrapolation_buckets.append(
            SyntheticTreeDepthBucket(
                depth=depth,
                samples=samples,
            )
        )

    return SyntheticTreeBenchmarkSplits(
        train_samples=tuple(train_samples),
        validation_samples=tuple(validation_samples),
        extrapolation_buckets=tuple(extrapolation_buckets),
    )


def _write_tree_samples_jsonl(
    path: Path,
    samples: tuple[SyntheticTreeTextSample, ...],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_serializable(), sort_keys=True))
            handle.write("\n")


def write_synthetic_tree_benchmark_artifacts(
    config: SyntheticTreeBenchmarkConfig,
    output_dir: str | Path,
) -> SyntheticTreeBenchmarkArtifacts:
    splits = build_synthetic_tree_benchmark(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / "benchmark_config.json"
    manifest_path = output_path / "benchmark_manifest.json"
    train_path = output_path / "train.jsonl"
    validation_path = output_path / "validation.jsonl"

    config_path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_tree_samples_jsonl(train_path, splits.train_samples)
    _write_tree_samples_jsonl(validation_path, splits.validation_samples)

    extrapolation_paths: list[tuple[int, str]] = []
    extrapolation_files: dict[str, str] = {}
    extrapolation_sample_counts: dict[str, int] = {}
    for bucket in splits.extrapolation_buckets:
        bucket_path = output_path / f"extrapolation_depth_{bucket.depth}.jsonl"
        _write_tree_samples_jsonl(bucket_path, bucket.samples)
        extrapolation_paths.append((bucket.depth, str(bucket_path)))
        extrapolation_files[str(bucket.depth)] = bucket_path.name
        extrapolation_sample_counts[str(bucket.depth)] = len(bucket.samples)

    manifest = {
        "config": asdict(config),
        "files": {
            "config": config_path.name,
            "manifest": manifest_path.name,
            "train": train_path.name,
            "validation": validation_path.name,
            "extrapolation": extrapolation_files,
        },
        "sample_counts": {
            "train": len(splits.train_samples),
            "validation": len(splits.validation_samples),
            "extrapolation": extrapolation_sample_counts,
        },
        "depth_sets": {
            "train": sorted({sample.depth for sample in splits.train_samples}),
            "validation": sorted({sample.depth for sample in splits.validation_samples}),
            "extrapolation": [bucket.depth for bucket in splits.extrapolation_buckets],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return SyntheticTreeBenchmarkArtifacts(
        output_dir=str(output_path),
        config_path=str(config_path),
        manifest_path=str(manifest_path),
        train_path=str(train_path),
        validation_path=str(validation_path),
        extrapolation_paths=tuple(extrapolation_paths),
    )
