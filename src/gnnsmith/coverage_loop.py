"""Python 3.12+ `sys.monitoring` zero-overhead coverage tracker + WhiteFox uncovered-branch finder."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import sys
from types import CodeType
from typing import Any, Callable, Dict, List, Set, Tuple

import torch_geometric

from gnnsmith.whitefox.inspector import SourceBranchTarget, extract_branch_targets


@dataclass
class UncoveredBranchesReport:
    files_executed: int
    lines_executed: int
    total_ast_branches_scanned: int
    uncovered_branches: List[SourceBranchTarget]


class PyGCoverageMonitor:
    """Tracks executed `(filepath, lineno)` pairs inside `torch_geometric` via `sys.monitoring`."""

    TOOL_ID = 4  # Dedicated sys.monitoring slot

    def __init__(self) -> None:
        self.pyg_root = str(Path(torch_geometric.__file__).resolve().parent)
        self.hit_lines: Dict[str, Set[int]] = {}
        self._active = False

    def _on_py_start(self, code: CodeType, instruction_offset: int):
        filename = code.co_filename
        if not filename.startswith(self.pyg_root):
            return sys.monitoring.DISABLE
        sys.monitoring.set_local_events(self.TOOL_ID, code, sys.monitoring.events.LINE)

    def _on_line(self, code: CodeType, line_number: int):
        self.hit_lines.setdefault(code.co_filename, set()).add(line_number)
        return sys.monitoring.DISABLE  # One-shot per line -> near-zero runtime overhead!

    def __enter__(self) -> "PyGCoverageMonitor":
        if sys.monitoring.get_tool(self.TOOL_ID) is not None:
            sys.monitoring.free_tool_id(self.TOOL_ID)
        sys.monitoring.use_tool_id(self.TOOL_ID, "gnnsmith_cov")
        sys.monitoring.register_callback(self.TOOL_ID, sys.monitoring.events.PY_START, self._on_py_start)
        sys.monitoring.register_callback(self.TOOL_ID, sys.monitoring.events.LINE, self._on_line)
        sys.monitoring.set_events(self.TOOL_ID, sys.monitoring.events.PY_START)
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._active:
            sys.monitoring.set_events(self.TOOL_ID, 0)
            sys.monitoring.register_callback(self.TOOL_ID, sys.monitoring.events.PY_START, None)
            sys.monitoring.register_callback(self.TOOL_ID, sys.monitoring.events.LINE, None)
            sys.monitoring.free_tool_id(self.TOOL_ID)
            self._active = False

    def find_uncovered_branches(self, modules: List[Any]) -> UncoveredBranchesReport:
        all_targets: List[SourceBranchTarget] = []
        uncovered: List[SourceBranchTarget] = []

        for mod in modules:
            try:
                mod_file = str(Path(inspect.getsourcefile(mod)).resolve())
            except Exception:
                continue
            hits = self.hit_lines.get(mod_file, set())
            targets = extract_branch_targets(mod)
            all_targets.extend(targets)
            for t in targets:
                # A branch's body wasn't entered if neither lineno+1 nor lineno+2 was executed
                if (t.lineno + 1) not in hits and (t.lineno + 2) not in hits:
                    uncovered.append(t)

        return UncoveredBranchesReport(
            files_executed=len(self.hit_lines),
            lines_executed=sum(len(s) for s in self.hit_lines.values()),
            total_ast_branches_scanned=len(all_targets),
            uncovered_branches=uncovered,
        )
