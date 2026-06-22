"""
ProposalState — LangGraph Runtime State Definition
====================================================

This TypedDict defines the shape of the state dictionary that LangGraph passes
between every node in the proposal-generation graph.

Key Design Decisions:
  - TypedDict (not Pydantic) because LangGraph's StateGraph expects a dict-like
    schema with annotated keys, not a Pydantic model.
  - All fields default to sensible empty values so a partially initialized state
    never causes KeyError downstream.
  - `shared_memory_path` is stored here so that ANY downstream node can read/
    write the persistent JSON without re-deriving the path.
"""

from __future__ import annotations

from typing import TypedDict, Optional, Dict


class SectionEntry(TypedDict):
    """Represents one section inside the shared memory file."""
    content: str
    status: str  # 'EMPTY' | 'DRAFT' | 'REVIEW' | 'FINAL'


class ProposalState(TypedDict, total=False):
    """
    LangGraph runtime state that flows through every node.

    Attributes
    ----------
    project_id : str
        Unique identifier for this tender/project. Used to resolve the
        project directory under ``storage/project_{project_id}/``.
    tender_text : str
        Raw extracted text from the Tender RFP PDF (~100 pages).
    company_assets_text : str
        Raw extracted text from the Company Profile / Past Experience
        document (PDF or Word).
    bid_details_text : str
        Raw text from the specific Tender/Bid details file (Markdown/Text).
    shared_memory_path : str
        Absolute path to the ``shared_memory.json`` file for this project.
        Stored in state so downstream nodes can read/write without path math.
    sections : Dict[str, SectionEntry]
        In-memory mirror of the 11 proposal sections. Updated after every
        writer node and flushed back to ``shared_memory.json``.
    current_section : Optional[str]
        The section key that the Universal_Writer_Node should draft next.
    error : Optional[str]
        If a node encounters a recoverable error, it sets this field instead
        of raising — the orchestrator can then decide to retry or abort.
    """

    project_id: str
    tender_text: str
    company_assets_text: str
    bid_details_text: str
    shared_memory_path: str
    sections: Dict[str, SectionEntry]
    current_section: Optional[str]
    output_markdown: Optional[str]
    error: Optional[str]
    force_reset: bool
    additional_assets_text: str
    language: Optional[str]  # 'ar' (Arabic) or 'en' (English) — defaults to 'ar'
    input_tokens: Optional[int]
    output_tokens: Optional[int]
