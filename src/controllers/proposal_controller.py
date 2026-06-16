"""
Proposal Generation Controller
==============================

Unified endpoint that merges file upload + Context_Initializer_Node execution
into a single API call, and provides on-demand section generation via the
Universal_Writer_Node.

Endpoints:
  ``POST /proposals/initialize/{project_id}``
  ``POST /proposals/generate/{project_id}/{section_type}``
  ``GET  /proposals/sections/{project_id}``
  ``GET  /proposals/sections/{project_id}/{section_type}``

Flow:
  1. Accept **one or more** files via multipart ``files`` field.
  2. Validate every file (extension, MIME, size).
  3. Save them into ``Assets/files/{project_id}/`` — creates the folder
     if it doesn't exist, or adds to it if it already does.
  4. Run the ``Context_Initializer_Node`` which discovers the files by
     keyword-matching on filenames (e.g. "tender", "company", "bid").
  5. Return the node output summary.

File Naming:
  Files are saved with their **original sanitized name** (no UUID suffix)
  so the node's keyword discovery works reliably.  If a file with the same
  name already exists in the project folder, it is **overwritten** — this
  lets the user re-upload a corrected version without leftover duplicates.
"""

from __future__ import annotations

import os
import aiofiles  # pyrefly: ignore[untyped-import]
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import ValidationError

from helpers.config import settings
from helpers.shared_memory import load_shared_memory, PROPOSAL_SECTIONS
from models.file_validation import FileValidationSchema
from models.proposal_state import ProposalState
from nodes.context_initializer import context_initializer_node
from nodes.universal_writer import universal_writer_node
from nodes.prompts_config import SECTIONS_CONFIG
from controllers.file_controller import sanitize_project_id

router = APIRouter(prefix="/proposals", tags=["Proposals"])

# Storage root — same convention used by all nodes
_STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename: strip directory components, keep only safe characters.
    Preserves the original name structure so the node's keyword-based
    discovery (e.g. "tender" in filename) still works.
    """
    base_name = os.path.basename(filename)
    name, ext = os.path.splitext(base_name)

    # Keep alphanumeric, dashes, underscores, dots (in the stem)
    clean_name = "".join(
        c for c in name if c.isalnum() or c in ("-", "_")
    ).strip()

    if not clean_name:
        clean_name = "uploaded_file"

    return f"{clean_name}{ext.lower()}"


def _validate_upload(file: UploadFile) -> FileValidationSchema:
    """Validate a single file's metadata (extension, MIME, size)."""
    file_size = file.size
    if file_size is None:
        try:
            file.file.seek(0, os.SEEK_END)
            file_size = file.file.tell()
            file.file.seek(0)
        except Exception:
            file_size = 0

    try:
        return FileValidationSchema(
            filename=file.filename or "",
            content_type=file.content_type or "",
            size=file_size,
        )
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = err.get("loc", [])
            msg = err.get("msg", "")
            if msg.startswith("Value error, "):
                msg = msg[13:]
            field = loc[0] if loc else "file"
            errors.append(f"{field}: {msg}")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": f"Validation failed for file '{file.filename}'.",
                "errors": errors,
            },
        )


async def _save_file(file: UploadFile, dest_path: Path) -> None:
    """Save an UploadFile to disk in 1 MB chunks."""
    try:
        async with aiofiles.open(dest_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                await out.write(chunk)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving '{file.filename}': {exc}",
        )
    finally:
        await file.close()


# ---------------------------------------------------------------------------
# Endpoint: Initialize Project
# ---------------------------------------------------------------------------

@router.post("/initialize/{project_id}", status_code=status.HTTP_200_OK)
async def initialize_proposal(
    project_id: str,
    force_reset: bool = False,
    file: Optional[List[UploadFile]] = File(None, description="One or more files uploaded under key 'file'"),
    files: Optional[List[UploadFile]] = File(None, description="One or more files uploaded under key 'files'"),
    tender_file: Optional[UploadFile] = File(None, description="Tender RFP document"),
    company_profile: Optional[UploadFile] = File(None, description="Company Profile / Experience document"),
    bid_details: Optional[UploadFile] = File(None, description="Specific Bid details document"),
):
    """
    Upload one or more files for a project and initialize the context.

    - Creates ``Assets/files/{project_id}/`` if it doesn't exist.
    - If the folder already exists, new files are **added** to it
      (existing files with the same name are overwritten).
    - After saving, the ``Context_Initializer_Node`` runs automatically:
      it extracts text from all discovered documents and creates
      ``shared_memory.json`` in ``storage/project_{project_id}/``.

    **File naming convention:** include a keyword in the filename so the
    node knows which role each file plays:

    | Role | Keywords in filename |
    |------|---------------------|
    | Tender RFP | ``tender``, ``rfp`` |
    | Company Profile | ``company``, ``profile``, ``experience`` |
    | Bid Details | ``bid``, ``details`` |
    """
    # 1. Sanitize project ID
    clean_project_id = sanitize_project_id(project_id)

    # Collect all uploaded files dynamically
    uploaded_files_list: List[UploadFile] = []
    if file:
        uploaded_files_list.extend(file)
    if files:
        uploaded_files_list.extend(files)
    if tender_file:
        uploaded_files_list.append(tender_file)
    if company_profile:
        uploaded_files_list.append(company_profile)
    if bid_details:
        uploaded_files_list.append(bid_details)

    if not uploaded_files_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "No files uploaded.",
                "errors": ["At least one file must be provided using 'file', 'files', 'tender_file', 'company_profile', or 'bid_details' keys."]
            }
        )

    # 2. Validate ALL files before writing anything to disk
    validated: List[tuple[UploadFile, FileValidationSchema]] = []
    for f in uploaded_files_list:
        schema = _validate_upload(f)
        validated.append((f, schema))

    # 3. Create / reuse the project assets directory
    project_assets_dir = settings.assets_dir / clean_project_id
    project_assets_dir.mkdir(parents=True, exist_ok=True)

    # 4. Save each file with its original (sanitized) name
    saved_files: List[Dict[str, Any]] = []
    for f, schema in validated:
        safe_name = _sanitize_filename(schema.filename)
        dest_path = project_assets_dir / safe_name

        await _save_file(f, dest_path)

        saved_files.append({
            "original_filename": schema.filename,
            "saved_filename": safe_name,
            "content_type": schema.content_type,
            "size_bytes": schema.size,
        })

    # 5. Run the Context_Initializer_Node
    initial_state: ProposalState = {"project_id": clean_project_id, "force_reset": force_reset}
    node_result = context_initializer_node(initial_state)

    if "error" in node_result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=node_result["error"],
        )

    # 6. Build response
    return {
        "status": "success",
        "message": f"{len(saved_files)} file(s) uploaded and context initialized.",
        "project_id": clean_project_id,
        "uploaded_files": saved_files,
        "shared_memory_path": node_result.get("shared_memory_path"),
        "node_output": {
            "tender_text_length": len(node_result.get("tender_text", "")),
            "company_assets_text_length": len(node_result.get("company_assets_text", "")),
            "bid_details_text_length": len(node_result.get("bid_details_text", "")),
            "sections_initialized": list(node_result.get("sections", {}).keys()),
        },
    }


# ---------------------------------------------------------------------------
# Endpoint: Generate a Section (Universal_Writer_Node)
# ---------------------------------------------------------------------------

@router.post("/generate/{project_id}/{section_type}", status_code=status.HTTP_200_OK)
async def generate_section(project_id: str, section_type: str):
    """
    Generate a single proposal section using the Universal_Writer_Node.

    This endpoint is called **on-demand** by the frontend whenever the user
    wants to draft a specific section. It:

    1. Validates the project exists and has been initialized.
    2. Loads the shared_memory.json to get tender/company text and
       previously generated sections.
    3. Invokes Gemini via the Universal_Writer_Node.
    4. Returns the generated Markdown and persists it to the JSON file.

    **Path Parameters:**
      - ``project_id``: The project identifier (must match an initialized project).
      - ``section_type``: One of the 11 section keys (e.g. ``methodology``).

    **Valid section_type values:**
      ``cover_letter``, ``executive_summary``, ``scope_understanding``,
      ``vision_2030``, ``company_profile``, ``past_projects``,
      ``methodology``, ``team``, ``timeline``, ``quality_and_risk``,
      ``pricing``
    """
    # 1. Sanitize & validate inputs
    clean_project_id = sanitize_project_id(project_id)
    section_key = section_type.strip().lower()

    if section_key not in SECTIONS_CONFIG:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": f"Unknown section type: '{section_type}'.",
                "valid_sections": list(SECTIONS_CONFIG.keys()),
            },
        )

    # 2. Verify the project has been initialized (shared_memory.json exists)
    project_dir = _STORAGE_ROOT / f"project_{clean_project_id}"
    try:
        shared_memory = load_shared_memory(project_dir)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": f"Project '{clean_project_id}' has not been initialized.",
                "hint": "Call POST /proposals/initialize/{project_id} first.",
            },
        )

    # 3. Build the LangGraph state from the persisted shared memory
    #    The tender_text and company_assets_text were extracted during
    #    initialization — we need to reload them from the context_initializer
    #    output. Since we're file-based, we re-run the text extraction
    #    by invoking the context_initializer in read-only mode, or we
    #    store the extracted texts. For efficiency, we store them in
    #    shared_memory during initialization.
    #
    #    CURRENT APPROACH: Re-extract from the Assets folder to keep
    #    shared_memory.json focused on section content only.
    from nodes.context_initializer import context_initializer_node as _init_node

    # Re-run the initializer to get fresh extracted text
    # (it's idempotent — it just re-reads files and re-creates shared_memory)
    init_state: ProposalState = {"project_id": clean_project_id}
    init_result = _init_node(init_state)

    if "error" in init_result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload project context: {init_result['error']}",
        )

    # 4. Construct the writer node state
    writer_state: ProposalState = {
        "project_id": clean_project_id,
        "current_section": section_key,
        "tender_text": init_result.get("tender_text", ""),
        "company_assets_text": init_result.get("company_assets_text", ""),
        "bid_details_text": init_result.get("bid_details_text", ""),
        "additional_assets_text": init_result.get("additional_assets_text", ""),
        "shared_memory_path": init_result.get("shared_memory_path", ""),
        "sections": init_result.get("sections", {}),
    }

    # 5. Invoke the Universal_Writer_Node
    writer_result = universal_writer_node(writer_state)

    if "error" in writer_result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": f"Section generation failed for '{section_key}'.",
                "error": writer_result["error"],
            },
        )

    # 6. Return the generated content
    return {
        "status": "success",
        "project_id": clean_project_id,
        "section_type": section_key,
        "section_config_type": SECTIONS_CONFIG[section_key]["type"],
        "generated_markdown": writer_result.get("output_markdown", ""),
        "input_tokens": writer_result.get("input_tokens", 0),
        "output_tokens": writer_result.get("output_tokens", 0),
        "sections_progress": {
            key: {
                "status": val.get("status", "EMPTY"),
                "has_content": bool(val.get("content", "").strip()),
            }
            for key, val in writer_result.get("sections", {}).items()
        },
    }


# ---------------------------------------------------------------------------
# Endpoint: Get Sections Status / Content
# ---------------------------------------------------------------------------

@router.get("/sections/{project_id}", status_code=status.HTTP_200_OK)
async def get_all_sections(project_id: str):
    """
    Retrieve the status and content summary of all 11 sections for a project.

    Returns each section's status (EMPTY/DRAFT/REVIEW/FINAL) and whether
    it has generated content.
    """
    clean_project_id = sanitize_project_id(project_id)
    project_dir = _STORAGE_ROOT / f"project_{clean_project_id}"

    try:
        shared_memory = load_shared_memory(project_dir)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": f"Project '{clean_project_id}' has not been initialized.",
                "hint": "Call POST /proposals/initialize/{project_id} first.",
            },
        )

    sections = shared_memory.get("sections", {})

    return {
        "status": "success",
        "project_id": clean_project_id,
        "metadata": shared_memory.get("metadata", {}),
        "sections": {
            key: {
                "status": val.get("status", "EMPTY"),
                "has_content": bool(val.get("content", "").strip()),
                "content_length": len(val.get("content", "")),
            }
            for key, val in sections.items()
        },
    }


@router.get("/sections/{project_id}/{section_type}", status_code=status.HTTP_200_OK)
async def get_section_content(project_id: str, section_type: str):
    """
    Retrieve the full generated content for a specific section.

    Returns the Markdown content and status of the requested section.
    """
    clean_project_id = sanitize_project_id(project_id)
    section_key = section_type.strip().lower()

    if section_key not in PROPOSAL_SECTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": f"Unknown section type: '{section_type}'.",
                "valid_sections": PROPOSAL_SECTIONS,
            },
        )

    project_dir = _STORAGE_ROOT / f"project_{clean_project_id}"

    try:
        shared_memory = load_shared_memory(project_dir)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": f"Project '{clean_project_id}' has not been initialized.",
                "hint": "Call POST /proposals/initialize/{project_id} first.",
            },
        )

    sections = shared_memory.get("sections", {})
    section_data = sections.get(section_key, {"content": "", "status": "EMPTY"})

    return {
        "status": "success",
        "project_id": clean_project_id,
        "section_type": section_key,
        "section_status": section_data.get("status", "EMPTY"),
        "content": section_data.get("content", ""),
        "content_length": len(section_data.get("content", "")),
    }

