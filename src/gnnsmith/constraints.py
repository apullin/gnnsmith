"""Z3 SMT constraint context for solving symbolic GNN DAG dimensions and parameters."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import z3


class Z3GraphContext:
    """Manages Z3 symbolic variables and constraints during GNN DAG generation."""

    def __init__(self, rng: random.Random, timeout_ms: int = 1500) -> None:
        self.rng = rng
        self.solver = z3.Solver()
        self.solver.set("timeout", timeout_ms)
        self._sym_counter = 0
        self.symbols: Dict[str, z3.ArithRef] = {}

    def new_int(self, prefix: str, low: int = 1, high: int = 64) -> z3.ArithRef:
        name = f"{prefix}_{self._sym_counter}"
        self._sym_counter += 1
        sym = z3.Int(name)
        self.symbols[name] = sym
        self.solver.add(sym >= low, sym <= high)
        # Add soft diversity nudges by pinning random lower/mid/upper targets via push-friendly assumptions
        return sym

    def require(self, *constraints: z3.BoolRef | bool) -> None:
        for c in constraints:
            if isinstance(c, bool):
                if not c:
                    self.solver.add(z3.BoolVal(False))
            else:
                self.solver.add(c)

    def check_with(self, *extra_constraints: z3.BoolRef | bool) -> bool:
        self.solver.push()
        try:
            self.require(*extra_constraints)
            return self.solver.check() == z3.sat
        finally:
            self.solver.pop()

    def try_commit(self, *constraints: z3.BoolRef | bool) -> bool:
        self.solver.push()
        self.require(*constraints)
        if self.solver.check() == z3.sat:
            # Keep constraints by popping and re-adding cleanly at current level
            self.solver.pop()
            self.require(*constraints)
            return True
        self.solver.pop()
        return False

    def solve(self, randomize_hints: Optional[List[Tuple[z3.ArithRef, int]]] = None) -> Optional[z3.ModelRef]:
        """Solve constraints, optionally trying random concrete hints first for diversity."""
        if randomize_hints:
            self.solver.push()
            for sym, val in randomize_hints:
                self.solver.add(sym == val)
            if self.solver.check() == z3.sat:
                model = self.solver.model()
                self.solver.pop()
                return model
            self.solver.pop()

        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None

    @staticmethod
    def eval_int(model: z3.ModelRef, expr: int | z3.ArithRef) -> int:
        if isinstance(expr, int):
            return expr
        val = model.eval(expr, model_completion=True)
        return int(val.as_long())
