"""AST-based white-box branch inspector for PyTorch Geometric modules.

Implements the WhiteFox stage-1 concept programmatically: parses installed PyG source
files into Python ASTs and extracts branch predicates (`if` conditions, `@implements`
dispatch targets, `NotImplementedError` / `assert` guards, device/compile fast-paths)
to guide targeted test synthesis and track branch-target inventory.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import Any, List


@dataclass
class SourceBranchTarget:
    module_name: str
    qualname: str
    lineno: int
    condition_src: str
    has_device_branch: bool
    has_compile_branch: bool
    has_grad_branch: bool
    has_ptr_or_sort_branch: bool


def extract_branch_targets(py_module: Any) -> List[SourceBranchTarget]:
    """Extract conditional branch predicates from a Python module's AST."""
    try:
        src = inspect.getsource(py_module)
    except OSError:
        return []

    tree = ast.parse(src)
    targets: List[SourceBranchTarget] = []
    mod_name = getattr(py_module, "__name__", str(py_module))

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope_stack: List[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope_stack.append(node.name)
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.scope_stack.append(node.name)
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_If(self, node: ast.If) -> None:
            cond = ast.unparse(node.test)
            low = cond.lower()
            targets.append(
                SourceBranchTarget(
                    module_name=mod_name,
                    qualname=".".join(self.scope_stack) or "<module>",
                    lineno=node.lineno,
                    condition_src=cond,
                    has_device_branch=any(k in low for k in ("is_cuda", "device", "mps", "cpu")),
                    has_compile_branch=any(k in low for k in ("is_compiling", "onnx", "jit")),
                    has_grad_branch="requires_grad" in low or "needs_input_grad" in low,
                    has_ptr_or_sort_branch=any(k in low for k in ("ptr", "is_sorted", "sort_order", "transpose", "undirected")),
                )
            )
            self.generic_visit(node)

    _Visitor().visit(tree)
    return targets
