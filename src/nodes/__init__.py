"""
nodes — LangGraph Graph Nodes
==============================

Each file in this package contains one or more LangGraph node functions.
Nodes are pure functions with the signature:

    def node_name(state: ProposalState) -> dict

They receive the current graph state and return a partial dict of updates.
"""

from nodes.context_initializer import context_initializer_node
from nodes.universal_writer import universal_writer_node, universal_writer_stream
from nodes.prompts_config import SECTIONS_CONFIG, get_section_config, get_all_section_keys

__all__ = [
    "context_initializer_node",
    "universal_writer_node",
    "universal_writer_stream",
    "SECTIONS_CONFIG",
    "get_section_config",
    "get_all_section_keys",
]

