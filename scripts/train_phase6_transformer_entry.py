"""Lightweight entrypoint for Windows-safe Phase 6 trainer execution.

Running this script keeps multiprocessing worker bootstrap imports cheap on Windows,
while delegating full training logic to train_phase6_transformer.main().
"""

from __future__ import annotations

import os
import sys


def _arg_value(flag: str) -> str | None:
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


if __name__ == "__main__":
    model_variant = _arg_value("--model-variant")
    gnn_graph_variant = _arg_value("--gnn-graph-variant")

    if model_variant:
        os.environ["HALO_PHASE6_MODEL_VARIANT"] = model_variant
    if gnn_graph_variant:
        os.environ["HALO_PHASE6_GNN_GRAPH_VARIANT"] = gnn_graph_variant

    from train_phase6_transformer import main

    main()
