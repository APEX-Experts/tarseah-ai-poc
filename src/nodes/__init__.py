"""
nodes — LangGraph Graph Nodes
==============================

Each file in this package contains one or more LangGraph node functions.
Nodes are pure functions with the signature:

    def node_name(state: ProposalState) -> dict

They receive the current graph state and return a partial dict of updates.
"""

from nodes.context_initializer import context_initializer_node

__all__ = ["context_initializer_node"]
