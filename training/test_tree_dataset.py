import random
import unittest

from pascnn.data import SyntheticTreeBenchmarkConfig
from pascnn.data import build_synthetic_tree_benchmark
from pascnn.data import generate_synthetic_tree_exact_depth
from pascnn.data import generate_unique_synthetic_tree_samples
from pascnn.data import serialize_synthetic_tree
from pascnn.data import validate_synthetic_tree


class TreeDatasetTests(unittest.TestCase):
    def test_exact_depth_generator_respects_requested_depth(self) -> None:
        rng = random.Random(7)

        for depth in range(5):
            with self.subTest(depth=depth):
                tree = generate_synthetic_tree_exact_depth(depth, rng)
                canonical_text = serialize_synthetic_tree(tree)
                report = validate_synthetic_tree(canonical_text)
                self.assertTrue(report.is_valid)
                self.assertEqual(report.depth, depth)
                self.assertEqual(report.canonical_text, canonical_text)

    def test_unique_samples_match_requested_depth_and_are_distinct(self) -> None:
        rng = random.Random(11)

        samples = generate_unique_synthetic_tree_samples(
            depth=3,
            num_samples=10,
            rng=rng,
        )

        self.assertEqual(len(samples), 10)
        self.assertEqual(len({sample.text for sample in samples}), 10)
        self.assertTrue(all(sample.depth == 3 for sample in samples))
        self.assertTrue(all(validate_synthetic_tree(sample.text).is_valid for sample in samples))

    def test_benchmark_builder_keeps_depth_buckets_and_split_disjointness(self) -> None:
        config = SyntheticTreeBenchmarkConfig(
            train_max_depth=3,
            train_samples_per_depth=2,
            validation_samples_per_depth=1,
            extrapolation_depths=(5, 7),
            extrapolation_samples_per_depth=3,
            seed=19,
        )

        splits = build_synthetic_tree_benchmark(config)

        train_texts = {sample.text for sample in splits.train_samples}
        validation_texts = {sample.text for sample in splits.validation_samples}
        extrapolation_texts = {
            sample.text
            for bucket in splits.extrapolation_buckets
            for sample in bucket.samples
        }

        self.assertEqual(
            {sample.depth for sample in splits.train_samples},
            {0, 1, 2, 3},
        )
        self.assertEqual(
            {sample.depth for sample in splits.validation_samples},
            {0, 1, 2, 3},
        )
        self.assertEqual(
            tuple(bucket.depth for bucket in splits.extrapolation_buckets),
            (5, 7),
        )
        self.assertTrue(train_texts.isdisjoint(validation_texts))
        self.assertTrue(train_texts.isdisjoint(extrapolation_texts))
        self.assertTrue(validation_texts.isdisjoint(extrapolation_texts))
        for bucket in splits.extrapolation_buckets:
            self.assertTrue(all(sample.depth == bucket.depth for sample in bucket.samples))

    def test_benchmark_builder_is_reproducible_for_fixed_seed(self) -> None:
        config = SyntheticTreeBenchmarkConfig(
            train_max_depth=2,
            train_samples_per_depth=2,
            validation_samples_per_depth=1,
            extrapolation_depths=(4,),
            extrapolation_samples_per_depth=2,
            seed=23,
        )

        splits_a = build_synthetic_tree_benchmark(config)
        splits_b = build_synthetic_tree_benchmark(config)

        self.assertEqual(splits_a.to_serializable(), splits_b.to_serializable())

    def test_benchmark_builder_rejects_impossible_depth_zero_capacity(self) -> None:
        config = SyntheticTreeBenchmarkConfig(
            train_max_depth=1,
            train_samples_per_depth=2,
            validation_samples_per_depth=2,
            extrapolation_depths=(3,),
            extrapolation_samples_per_depth=2,
            seed=29,
        )

        with self.assertRaisesRegex(
            ValueError,
            "but only 3 unique trees exist.",
        ):
            build_synthetic_tree_benchmark(config)

    def test_benchmark_builder_supports_explicit_train_and_validation_depth_sets(self) -> None:
        config = SyntheticTreeBenchmarkConfig(
            train_max_depth=3,
            train_samples_per_depth=5,
            validation_samples_per_depth=2,
            train_depths=(2, 3),
            validation_depths=(2, 3),
            extrapolation_depths=(4, 5),
            extrapolation_samples_per_depth=4,
            seed=31,
        )

        splits = build_synthetic_tree_benchmark(config)

        self.assertEqual({sample.depth for sample in splits.train_samples}, {2, 3})
        self.assertEqual({sample.depth for sample in splits.validation_samples}, {2, 3})
        self.assertEqual(
            tuple(bucket.depth for bucket in splits.extrapolation_buckets),
            (4, 5),
        )
        self.assertEqual(len(splits.train_samples), 10)
        self.assertEqual(len(splits.validation_samples), 4)

    def test_benchmark_builder_defaults_validation_depths_to_train_depths(self) -> None:
        config = SyntheticTreeBenchmarkConfig(
            train_max_depth=3,
            train_samples_per_depth=4,
            validation_samples_per_depth=2,
            train_depths=(2, 3),
            extrapolation_depths=(4,),
            extrapolation_samples_per_depth=3,
            seed=37,
        )

        splits = build_synthetic_tree_benchmark(config)

        self.assertEqual({sample.depth for sample in splits.train_samples}, {2, 3})
        self.assertEqual({sample.depth for sample in splits.validation_samples}, {2, 3})

    def test_benchmark_builder_rejects_extrapolation_depths_not_above_train_and_validation(self) -> None:
        config = SyntheticTreeBenchmarkConfig(
            train_max_depth=3,
            train_samples_per_depth=4,
            validation_samples_per_depth=2,
            train_depths=(2, 3),
            validation_depths=(2, 3),
            extrapolation_depths=(3, 4),
            extrapolation_samples_per_depth=3,
            seed=41,
        )

        with self.assertRaisesRegex(
            ValueError,
            "every extrapolation depth must be greater than every train/validation depth.",
        ):
            build_synthetic_tree_benchmark(config)


if __name__ == "__main__":
    unittest.main()
