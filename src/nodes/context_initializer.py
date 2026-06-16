"""
Context_Initializer_Node — The First Node in the Proposal Generation Graph
===========================================================================

This is the ENTRY POINT of the LangGraph pipeline. It runs once per project
and performs the following orchestration:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. INITIALIZE DIRECTORY                                           │
    │     Ensure ``storage/project_{project_id}/`` exists.              │
    │                                                                    │
    │  2. TEXT EXTRACTION                                                 │
    │     Read 3 uploaded input files from the project's Assets folder:  │
    │       • Tender RFP (PDF, ~100 pages)                              │
    │       • Company Profile / Past Experience (PDF or DOCX)           │
    │       • Bid Details (Markdown or Text)                            │
    │     Extract raw text using modular extractors (helpers/text_ext…). │
    │                                                                    │
    │  3. INITIALIZE shared_memory.json                                  │
    │     Create the persistent JSON file inside the project directory   │
    │     with 11 empty proposal sections, each status = "EMPTY".       │
    │                                                                    │
    │  4. RETURN STATE UPDATE                                            │
    │     Return a partial dict to LangGraph containing:                │
    │       - project_id                                                 │
    │       - tender_text                                                │
    │       - company_assets_text                                        │
    │       - bid_details_text                                           │
    │       - shared_memory_path                                         │
    │       - sections (in-memory mirror)                               │
    └─────────────────────────────────────────────────────────────────────┘

How the Next Node Reads This
-----------------------------
The ``Universal_Writer_Node`` (next in the graph) will receive the updated
``ProposalState`` with all extracted text pre-loaded. It can also call
``load_shared_memory(project_dir)`` at any time to read the persistent JSON
and write individual sections back via ``update_section()``.

File Discovery Strategy
-----------------------
Input files are expected to be uploaded to the project's Assets folder
(``src/Assets/files/{project_id}/``) by the existing file upload API.
This node looks for files whose names contain role keywords:
  - ``"tender"`` or ``"rfp"``  → Tender RFP
  - ``"company"`` or ``"profile"`` or ``"experience"``  → Company Profile
  - ``"bid"`` or ``"details"``  → Bid Details

If a file cannot be found by keyword, the node falls back to a positional
strategy and logs a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from models.proposal_state import ProposalState
from helpers.text_extraction import extract_text, find_file_by_role
from helpers.shared_memory import (
    initialize_shared_memory,
    load_shared_memory,
    build_default_sections,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base storage directory where project folders are created.
# Resolved relative to the src/ directory (where the app runs from).
_STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"

# The Assets directory where the file upload API stores uploaded documents.
_ASSETS_ROOT = Path(__file__).resolve().parent.parent / "Assets" / "files"


# ---------------------------------------------------------------------------
# File Discovery Helpers
# ---------------------------------------------------------------------------

def _discover_input_files(project_id: str) -> Dict[str, Path]:
    """
    Locate the three required input files inside the project's Assets folder.

    Search Strategy:
      1. Look for files whose name contains known role keywords.
      2. If a role keyword is not found, log a warning and skip.

    Parameters
    ----------
    project_id : str
        The sanitized project identifier.

    Returns
    -------
    dict
        Mapping of role → file path. Keys: ``"tender"``, ``"company"``,
        ``"bid"``. A key is absent if no matching file was found.
    """
    assets_dir = _ASSETS_ROOT / project_id

    if not assets_dir.exists():
        logger.warning(
            "Assets directory does not exist: '%s'. "
            "Make sure files are uploaded via the /files/upload/{project_id} API.",
            assets_dir,
        )
        return {}

    discovered: Dict[str, Path] = {}

    # --- Tender RFP ---
    tender_keywords = ["tender", "rfp", "كراسة", "شروط", "مواصفات"]
    for keyword in tender_keywords:
        match = find_file_by_role(assets_dir, keyword, [".pdf", ".docx", ".doc"])
        if match:
            discovered["tender"] = match
            break

    # --- Company Profile / Past Experience ---
    company_keywords = ["company", "profile", "experience", "شركة", "ملف", "خبرة", "سوابق"]
    for keyword in company_keywords:
        match = find_file_by_role(assets_dir, keyword, [".pdf", ".docx", ".doc"])
        if match:
            discovered["company"] = match
            break

    # --- Bid Details ---
    bid_keywords = ["bid", "details", "عرض", "تفاصيل"]
    for keyword in bid_keywords:
        match = find_file_by_role(assets_dir, keyword, [".md", ".txt", ".pdf", ".docx"])
        if match:
            discovered["bid"] = match
            break

    # Log discovery results
    for role in ["tender", "company", "bid"]:
        if role in discovered:
            logger.info("Discovered %s file: %s", role, discovered[role].name)
        else:
            logger.info(
                "No %s file found in '%s'.",
                role,
                assets_dir,
            )

    return discovered


# ---------------------------------------------------------------------------
# Main Node Function
# ---------------------------------------------------------------------------

def context_initializer_node(state: ProposalState) -> Dict[str, Any]:
    """
    Context_Initializer_Node — LangGraph Node Function.

    This is the first node executed in the proposal generation graph.
    It sets up the project workspace and extracts all raw input text.

    Parameters
    ----------
    state : ProposalState
        The current LangGraph state. Must contain at least ``project_id``.

    Returns
    -------
    dict
        Partial state update containing:
          - ``project_id``
          - ``tender_text``
          - ``company_assets_text``
          - ``bid_details_text``
          - ``shared_memory_path``
          - ``sections``

    Raises
    ------
    ValueError
        If ``project_id`` is missing or empty in the state.
    """
    # ------------------------------------------------------------------
    # Step 0: Validate project_id
    # ------------------------------------------------------------------
    project_id = state.get("project_id", "").strip()
    if not project_id:
        error_msg = "project_id is required but was not provided in the state."
        logger.error(error_msg)
        return {"error": error_msg}

    logger.info("=" * 60)
    logger.info("Context_Initializer_Node — START (project: %s)", project_id)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Initialize Project Directory
    # ------------------------------------------------------------------
    project_dir = _STORAGE_ROOT / f"project_{project_id}"
    project_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Project directory ready: %s", project_dir)

    # ------------------------------------------------------------------
    # Step 2: Discover & Extract Text from Input Files
    # ------------------------------------------------------------------
    # Step 2: Discover & Extract Text from Input Files
    # ------------------------------------------------------------------
    assets_dir = _ASSETS_ROOT / project_id
    discovered_files = _discover_input_files(project_id)

    tender_text = ""
    company_assets_text = ""
    bid_details_text = ""
    additional_parts: list[str] = []

    # Map primary paths to skip them when collecting additional assets
    primary_paths = set(discovered_files.values())

    # --- Extract Tender RFP ---
    if "tender" in discovered_files:
        try:
            raw_tender = extract_text(discovered_files["tender"])
            tender_text = f"=== اسم الملف: {discovered_files['tender'].name} ===\n{raw_tender}"
            logger.info(
                "Tender text extracted: %d characters from '%s'.",
                len(tender_text),
                discovered_files["tender"].name,
            )
        except Exception as exc:
            logger.error("Failed to extract tender text: %s", exc)
            tender_text = f"[EXTRACTION ERROR] {exc}"
    else:
        logger.info("Tender RFP file not found — tender_text will be empty.")

    # --- Extract Company Profile / Past Experience ---
    if "company" in discovered_files:
        try:
            raw_company = extract_text(discovered_files["company"])
            company_assets_text = f"=== اسم الملف: {discovered_files['company'].name} ===\n{raw_company}"
            logger.info(
                "Company assets text extracted: %d characters from '%s'.",
                len(company_assets_text),
                discovered_files["company"].name,
            )
        except Exception as exc:
            logger.error("Failed to extract company assets text: %s", exc)
            company_assets_text = f"[EXTRACTION ERROR] {exc}"
    else:
        logger.info(
            "Company profile file not found — company_assets_text will be empty."
        )

    # --- Extract Bid Details ---
    if "bid" in discovered_files:
        try:
            raw_bid = extract_text(discovered_files["bid"])
            bid_details_text = f"=== اسم الملف: {discovered_files['bid'].name} ===\n{raw_bid}"
            logger.info(
                "Bid details text extracted: %d characters from '%s'.",
                len(bid_details_text),
                discovered_files["bid"].name,
            )
        except Exception as exc:
            logger.error("Failed to extract bid details text: %s", exc)
            bid_details_text = f"[EXTRACTION ERROR] {exc}"
    else:
        logger.info(
            "Bid details file not found — bid_details_text will be empty."
        )

    # --- Extract All Other Files (Additional Assets) ---
    if assets_dir.exists():
        for child in assets_dir.iterdir():
            if child.is_file() and child not in primary_paths:
                try:
                    text = extract_text(child)
                    if text.strip():
                        additional_parts.append(
                            f"=== اسم الملف: {child.name} ===\n{text}"
                        )
                except ValueError:
                    # Ignore unsupported file types like images gracefully
                    logger.info("Skipping text extraction for unsupported file type: %s", child.name)
                except Exception as exc:
                    logger.error("Failed to extract text from additional file '%s': %s", child.name, exc)
                    additional_parts.append(
                        f"=== اسم الملف: {child.name} ===\n[خطأ في استخراج النص: {exc}]"
                    )

    additional_assets_text = "\n\n".join(additional_parts)

    force_reset = state.get("force_reset", False)
    shared_memory_path = initialize_shared_memory(project_dir, force_reset=force_reset)
    logger.info("Shared memory initialized at: %s", shared_memory_path)

    # Load the actual (potentially merged/preserved) sections from the file
    try:
        shared_memory = load_shared_memory(project_dir)
        sections = shared_memory.get("sections", {})
    except Exception:
        sections = build_default_sections()

    # ------------------------------------------------------------------
    # Step 4: Return State Update
    # ------------------------------------------------------------------
    state_update: Dict[str, Any] = {
        "project_id": project_id,
        "tender_text": tender_text,
        "company_assets_text": company_assets_text,
        "bid_details_text": bid_details_text,
        "additional_assets_text": additional_assets_text,
        "shared_memory_path": str(shared_memory_path),
        "sections": sections,
    }

    logger.info("=" * 60)
    logger.info("Context_Initializer_Node — COMPLETE")
    logger.info(
        "  Tender: %d chars | Company: %d chars | Bid: %d chars | Additional: %d chars",
        len(tender_text),
        len(company_assets_text),
        len(bid_details_text),
        len(additional_assets_text),
    )
    logger.info("  Sections initialized: %d", len(sections))
    logger.info("  Shared memory: %s", shared_memory_path)
    logger.info("=" * 60)

    return state_update
