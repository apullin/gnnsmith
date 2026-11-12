"""Registry of all GNNSmith operators."""

from __future__ import annotations

from typing import Dict, List, Type

from gnnsmith.ops.base import AbsGNNOp
from gnnsmith.ops.pyg_conv import CONV_OPS
from gnnsmith.ops.pyg_norm_pool import COMBINATOR_OPS, NORM_OPS, POOL_OPS, GlobalPoolReadoutOp

ALL_OPS: List[Type[AbsGNNOp]] = [
    *CONV_OPS,
    *NORM_OPS,
    *POOL_OPS,
    *COMBINATOR_OPS,
    GlobalPoolReadoutOp,
]

OP_REGISTRY: Dict[str, Type[AbsGNNOp]] = {op.op_type: op for op in ALL_OPS}
