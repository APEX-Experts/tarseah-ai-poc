"""
Shared Memory Manager
=====================

Manages the ``shared_memory.json`` file that lives inside each project
directory (``storage/project_{project_id}/shared_memory.json``).

This file acts as the persistent "brain" of the proposal generation pipeline.
It stores the 11 default proposal sections with their content and status,
allowing any node or any future API call to pick up where the last one left off.

Why JSON and not SQLite?
------------------------
  - For a PoC without a vector database, a flat JSON file keeps things dead
    simple, human-readable, and debuggable.
  - Downstream nodes (Universal_Writer_Node, etc.) can read/update individual
    sections and flush back without any ORM overhead.

Section Lifecycle:
  EMPTY → DRAFT → REVIEW → FINAL
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The 11 Default Proposal Sections
# ---------------------------------------------------------------------------

PROPOSAL_SECTIONS: list[str] = [
    "cover_letter",
    "executive_summary",
    "scope_understanding",
    "vision_2030",
    "company_profile",
    "past_projects",
    "methodology",
    "team",
    "timeline",
    "quality_and_risk",
    "pricing",
]

SHARED_MEMORY_FILENAME = "shared_memory.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_default_sections() -> Dict[str, Dict[str, str]]:
    """
    Return a fresh dictionary of the 11 sections, each initialized to
    ``{"content": "", "status": "EMPTY", "summary": ""}``.

    Returns
    -------
    dict
        ``{ "cover_letter": {"content": "", "status": "EMPTY", "summary": ""}, ... }``
    """
    return {
        section: {"content": "", "status": "EMPTY", "summary": ""}
        for section in PROPOSAL_SECTIONS
    }


def initialize_shared_memory(project_dir: Path, force_reset: bool = False) -> Path:
    """
    Create (or initialize) the ``shared_memory.json`` file inside the given
    project directory with the default 11-section skeleton.

    Parameters
    ----------
    project_dir : Path
        The project directory (e.g. ``storage/project_abc123/``).
    force_reset : bool, optional
        If True, clears any existing section content/status.
        If False (default), preserves any previously generated section data.

    Returns
    -------
    Path
        Absolute path to the newly created ``shared_memory.json``.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    memory_path = project_dir / SHARED_MEMORY_FILENAME

    # Start with the default 11-section empty skeleton
    sections = build_default_sections()

    # If the file already exists and we are not forcing a reset, try to merge
    if memory_path.exists() and not force_reset:
        try:
            existing_data = _read_json(memory_path)
            existing_sections = existing_data.get("sections", {})
            for sec_key, sec_val in existing_sections.items():
                if sec_key in sections:
                    # Preserve section content and status if it exists
                    if sec_val.get("content") or sec_val.get("status") != "EMPTY":
                        sections[sec_key] = sec_val
            logger.info("Preserved existing sections from shared memory.")
        except Exception as e:
            logger.warning("Failed to read existing shared memory file: %s. Re-initializing.", e)

    completed = sum(
        1 for s in sections.values() if s.get("status") != "EMPTY"
    )

    payload: Dict[str, Any] = {
        "project_id": project_dir.name,
        "sections": sections,
        "metadata": {
            "version": "1.0",
            "total_sections": len(PROPOSAL_SECTIONS),
            "completed_sections": completed,
        },
    }

    _write_json(memory_path, payload)
    logger.info("Initialized shared_memory.json at '%s'.", memory_path)
    return memory_path


def load_shared_memory(project_dir: Path) -> Dict[str, Any]:
    """
    Load and return the contents of ``shared_memory.json`` for a project.

    Parameters
    ----------
    project_dir : Path
        The project directory.

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If ``shared_memory.json`` does not exist in the directory.
    """
    memory_path = Path(project_dir) / SHARED_MEMORY_FILENAME
    if not memory_path.exists():
        raise FileNotFoundError(
            f"shared_memory.json not found in '{project_dir}'. "
            "Run the Context_Initializer_Node first."
        )
    return _read_json(memory_path)


def update_section(
    project_dir: Path,
    section_key: str,
    content: str,
    status: str = "DRAFT",
    summary: str = "",
) -> None:
    """
    Update a single section in the ``shared_memory.json`` file.

    Parameters
    ----------
    project_dir : Path
        The project directory.
    section_key : str
        One of the 11 section keys (e.g. ``"executive_summary"``).
    content : str
        The generated content for this section.
    status : str
        New status — typically ``"DRAFT"``, ``"REVIEW"``, or ``"FINAL"``.
    summary : str
        A condensed bullet-point summary of the section content.


    Raises
    ------
    ValueError
        If ``section_key`` is not one of the known 11 sections.
    """
    if section_key not in PROPOSAL_SECTIONS:
        raise ValueError(
            f"Unknown section '{section_key}'. "
            f"Valid sections: {PROPOSAL_SECTIONS}"
        )

    memory = load_shared_memory(project_dir)
    
    # Preserve existing summary if one isn't provided
    existing_entry = memory["sections"].get(section_key, {})
    new_summary = summary if summary else existing_entry.get("summary", "")

    memory["sections"][section_key] = {
        "content": content,
        "status": status,
        "summary": new_summary,
    }

    # Update completed-sections count
    completed = sum(
        1 for s in memory["sections"].values() if s["status"] != "EMPTY"
    )
    memory["metadata"]["completed_sections"] = completed

    memory_path = Path(project_dir) / SHARED_MEMORY_FILENAME
    _write_json(memory_path, memory)
    logger.info(
        "Updated section '%s' → status='%s' (%d/%d complete).",
        section_key,
        status,
        completed,
        len(PROPOSAL_SECTIONS),
    )


def store_extracted_texts(
    project_dir: Path,
    tender_text: str = "",
    company_assets_text: str = "",
    bid_details_text: str = "",
    additional_assets_text: str = "",
) -> None:
    """
    Persist extracted document texts into ``shared_memory.json`` so they
    can be loaded directly without re-running the context initializer.

    Parameters
    ----------
    project_dir : Path
        The project directory.
    tender_text, company_assets_text, bid_details_text, additional_assets_text : str
        The extracted text from each document category.
    """
    memory = load_shared_memory(project_dir)
    memory["extracted_texts"] = {
        "tender_text": tender_text,
        "company_assets_text": company_assets_text,
        "bid_details_text": bid_details_text,
        "additional_assets_text": additional_assets_text,
    }
    memory_path = Path(project_dir) / SHARED_MEMORY_FILENAME
    _write_json(memory_path, memory)
    logger.info(
        "Stored extracted texts in shared_memory.json "
        "(tender=%d, company=%d, bid=%d, additional=%d chars).",
        len(tender_text),
        len(company_assets_text),
        len(bid_details_text),
        len(additional_assets_text),
    )


def load_extracted_texts(project_dir: Path) -> Dict[str, str] | None:
    """
    Load previously stored extracted texts from ``shared_memory.json``.

    Returns
    -------
    dict | None
        A dict with keys ``tender_text``, ``company_assets_text``,
        ``bid_details_text``, ``additional_assets_text``.
        Returns ``None`` if extracted texts have not been stored yet.
    """
    memory = load_shared_memory(project_dir)
    return memory.get("extracted_texts")


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file and return its parsed content."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write a dictionary to a JSON file with pretty formatting."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
