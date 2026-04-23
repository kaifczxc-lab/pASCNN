"""Canonical tree language oracle for structural extrapolation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


TREE_INTERNAL_LABELS: tuple[str, ...] = ("A", "B", "C")
TREE_LEAF_LABELS: tuple[str, ...] = ("x", "y", "z")


@dataclass(frozen=True)
class SyntheticTree:
    label: str
    children: tuple["SyntheticTree", ...] = ()

    def __post_init__(self) -> None:
        is_leaf = len(self.children) == 0
        if is_leaf:
            if self.label not in TREE_LEAF_LABELS:
                raise ValueError("leaf labels must be one of: x, y, z.")
            return
        if len(self.children) != 2:
            raise ValueError("internal nodes must have exactly two children.")
        if self.label not in TREE_INTERNAL_LABELS:
            raise ValueError("internal labels must be one of: A, B, C.")

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass(frozen=True)
class TreeValidationReport:
    is_valid: bool
    error: str | None
    depth: int | None
    node_count: int | None
    leaf_count: int | None
    canonical_text: str | None

    def to_serializable(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "error": self.error,
            "depth": self.depth,
            "node_count": self.node_count,
            "leaf_count": self.leaf_count,
            "canonical_text": self.canonical_text,
        }


class _TreeParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def parse(self) -> SyntheticTree:
        tree = self._parse_tree()
        self._skip_whitespace()
        if self.index != len(self.text):
            raise ValueError("unexpected trailing input.")
        return tree

    def _parse_tree(self) -> SyntheticTree:
        self._skip_whitespace()
        if self.index >= len(self.text):
            raise ValueError("unexpected end of input.")
        char = self.text[self.index]
        if char == "(":
            return self._parse_internal()
        if char in TREE_LEAF_LABELS:
            self.index += 1
            return SyntheticTree(label=char)
        raise ValueError(f"unexpected token '{char}'.")

    def _parse_internal(self) -> SyntheticTree:
        self._expect("(")
        self._skip_whitespace()
        label = self._parse_label()
        self._require_whitespace("expected whitespace after internal label.")
        left = self._parse_tree()
        self._require_whitespace("expected whitespace between internal children.")
        right = self._parse_tree()
        self._skip_whitespace()
        self._expect(")")
        return SyntheticTree(label=label, children=(left, right))

    def _parse_label(self) -> str:
        if self.index >= len(self.text):
            raise ValueError("unexpected end of input while reading label.")
        label = self.text[self.index]
        if label not in TREE_INTERNAL_LABELS:
            raise ValueError(f"unexpected internal label '{label}'.")
        self.index += 1
        return label

    def _require_whitespace(self, error_message: str) -> None:
        if self.index >= len(self.text) or not self.text[self.index].isspace():
            raise ValueError(error_message)
        self._skip_whitespace()

    def _skip_whitespace(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def _expect(self, token: str) -> None:
        if self.index >= len(self.text) or self.text[self.index] != token:
            raise ValueError(f"expected '{token}'.")
        self.index += 1


def parse_synthetic_tree(text: str) -> SyntheticTree:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return _TreeParser(text).parse()


def serialize_synthetic_tree(tree: SyntheticTree) -> str:
    if not isinstance(tree, SyntheticTree):
        raise TypeError("tree must be a SyntheticTree instance.")
    if tree.is_leaf:
        return tree.label
    left, right = tree.children
    return (
        f"({tree.label} "
        f"{serialize_synthetic_tree(left)} "
        f"{serialize_synthetic_tree(right)})"
    )


def synthetic_tree_depth(tree: SyntheticTree) -> int:
    if not isinstance(tree, SyntheticTree):
        raise TypeError("tree must be a SyntheticTree instance.")
    if tree.is_leaf:
        return 0
    return 1 + max(synthetic_tree_depth(child) for child in tree.children)


def synthetic_tree_node_count(tree: SyntheticTree) -> int:
    if not isinstance(tree, SyntheticTree):
        raise TypeError("tree must be a SyntheticTree instance.")
    return 1 + sum(synthetic_tree_node_count(child) for child in tree.children)


def synthetic_tree_leaf_count(tree: SyntheticTree) -> int:
    if not isinstance(tree, SyntheticTree):
        raise TypeError("tree must be a SyntheticTree instance.")
    if tree.is_leaf:
        return 1
    return sum(synthetic_tree_leaf_count(child) for child in tree.children)


def validate_synthetic_tree(text: str) -> TreeValidationReport:
    try:
        tree = parse_synthetic_tree(text)
    except (TypeError, ValueError) as error:
        return TreeValidationReport(
            is_valid=False,
            error=str(error),
            depth=None,
            node_count=None,
            leaf_count=None,
            canonical_text=None,
        )

    return TreeValidationReport(
        is_valid=True,
        error=None,
        depth=synthetic_tree_depth(tree),
        node_count=synthetic_tree_node_count(tree),
        leaf_count=synthetic_tree_leaf_count(tree),
        canonical_text=serialize_synthetic_tree(tree),
    )
