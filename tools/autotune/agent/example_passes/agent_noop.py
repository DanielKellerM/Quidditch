# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Example agent-authored xDSL pass.

Demonstrates the author -> register -> evaluate capability (author_pass.py): a
real ModulePass the agent wrote, loaded into the lowering pipeline. This one is a
no-op (semantics-preserving by construction) so it exercises the MECHANISM
safely; a genuine structural transform is authored exactly the same way
(subclass ModulePass, implement apply with RewritePatterns), and the autotuner's
Tier-1/Tier-2 gates keep it iff it compiles, stays correct, and is faster.
"""
from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass


class AgentNoopPass(ModulePass):
    name = "agent-noop"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        return
