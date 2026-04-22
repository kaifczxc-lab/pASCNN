"""Native extension loader namespace for pASCNN."""

from __future__ import annotations

import importlib
import json
from types import ModuleType

_CACHED_NATIVE_MODULE: ModuleType | None = None

_EXPECTED_OPS = (
    "soft_prefix_match",
    "complex_restrict_defect",
    "edge_coherence_uwca",
    "complex_scatter_add",
)


def load_native_extension(*, required: bool = True) -> ModuleType | None:
    global _CACHED_NATIVE_MODULE
    if _CACHED_NATIVE_MODULE is not None:
        return _CACHED_NATIVE_MODULE

    try:
        module = importlib.import_module("pascnn._C")
    except ImportError:
        if required:
            raise
        return None

    _CACHED_NATIVE_MODULE = module
    return module


def native_extension_status() -> dict[str, object]:
    module = load_native_extension(required=False)
    if module is None:
        return {
            "native_found": False,
            "native_loadable": False,
            "ops_registered": False,
            "module_name": "pascnn._C",
            "registered_ops": [],
        }

    op_namespace = getattr(__import__("torch"), "ops").pascnn
    registered_ops = [name for name in _EXPECTED_OPS if hasattr(op_namespace, name)]
    metadata: dict[str, object] = {}
    if hasattr(module, "extension_info"):
        try:
            metadata = json.loads(module.extension_info())
        except Exception:
            metadata = {"raw_extension_info": module.extension_info()}

    return {
        "native_found": True,
        "native_loadable": True,
        "ops_registered": len(registered_ops) == len(_EXPECTED_OPS),
        "module_name": module.__name__,
        "registered_ops": registered_ops,
        "metadata": metadata,
    }


__all__ = ["load_native_extension", "native_extension_status"]
