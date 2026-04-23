import unittest

from pascnn.data import SyntheticTree
from pascnn.data import parse_synthetic_tree
from pascnn.data import serialize_synthetic_tree
from pascnn.data import synthetic_tree_depth
from pascnn.data import synthetic_tree_leaf_count
from pascnn.data import synthetic_tree_node_count
from pascnn.data import validate_synthetic_tree


class TreeSynthTests(unittest.TestCase):
    def test_parse_and_serialize_roundtrip_on_canonical_tree(self) -> None:
        text = "(C (A x y) (B z x))"

        tree = parse_synthetic_tree(text)

        self.assertEqual(
            tree,
            SyntheticTree(
                label="C",
                children=(
                    SyntheticTree(
                        label="A",
                        children=(
                            SyntheticTree(label="x"),
                            SyntheticTree(label="y"),
                        ),
                    ),
                    SyntheticTree(
                        label="B",
                        children=(
                            SyntheticTree(label="z"),
                            SyntheticTree(label="x"),
                        ),
                    ),
                ),
            ),
        )
        self.assertEqual(serialize_synthetic_tree(tree), text)

    def test_parser_normalizes_whitespace_to_canonical_text(self) -> None:
        text = " (C\t(A x y)\n (B z x ) ) "

        report = validate_synthetic_tree(text)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.canonical_text, "(C (A x y) (B z x))")
        self.assertEqual(report.depth, 2)
        self.assertEqual(report.node_count, 7)
        self.assertEqual(report.leaf_count, 4)

    def test_depth_and_counts_match_manual_tree(self) -> None:
        tree = SyntheticTree(
            label="A",
            children=(
                SyntheticTree(label="x"),
                SyntheticTree(
                    label="B",
                    children=(
                        SyntheticTree(label="y"),
                        SyntheticTree(
                            label="C",
                            children=(
                                SyntheticTree(label="z"),
                                SyntheticTree(label="x"),
                            ),
                        ),
                    ),
                ),
            ),
        )

        self.assertEqual(synthetic_tree_depth(tree), 3)
        self.assertEqual(synthetic_tree_node_count(tree), 7)
        self.assertEqual(synthetic_tree_leaf_count(tree), 4)

    def test_validator_rejects_malformed_trees(self) -> None:
        malformed_inputs = (
            "(A x)",
            "(A x y",
            "(Q x y)",
            "(A x q)",
            "(A x y) trailing",
            "",
        )

        for text in malformed_inputs:
            with self.subTest(text=text):
                report = validate_synthetic_tree(text)
                self.assertFalse(report.is_valid)
                self.assertIsNotNone(report.error)
                self.assertIsNone(report.depth)
                self.assertIsNone(report.canonical_text)

    def test_synthetic_tree_rejects_invalid_node_shapes(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "internal nodes must have exactly two children.",
        ):
            SyntheticTree(
                label="A",
                children=(SyntheticTree(label="x"),),
            )

        with self.assertRaisesRegex(
            ValueError,
            "leaf labels must be one of: x, y, z.",
        ):
            SyntheticTree(label="A")


if __name__ == "__main__":
    unittest.main()
