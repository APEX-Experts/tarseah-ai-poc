"""
Smoke test for the Context_Initializer_Node.

Creates dummy input files (tender PDF, company DOCX, bid markdown),
uploads them to the Assets folder, then runs the node and asserts
the output state is correctly populated.

Run from project root:
    python -m pytest tests/test_context_initializer.py -v
"""

import json
import shutil
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixture: Create a temporary project with dummy files
# ---------------------------------------------------------------------------

TEST_PROJECT_ID = "test_smoke_001"
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
ASSETS_DIR = SRC_DIR / "Assets" / "files" / TEST_PROJECT_ID
STORAGE_DIR = SRC_DIR / "storage" / f"project_{TEST_PROJECT_ID}"


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """
    Before each test:  create the Assets directory with 3 dummy files.
    After each test:   clean up both Assets and storage directories.
    """
    # --- Setup ---
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dummy tender file (plain text pretending to be a .txt since we
    #    can't create real PDFs without reportlab; the extractor handles .txt)
    tender_path = ASSETS_DIR / "tender_rfp_document.txt"
    tender_path.write_text(
        "TENDER RFP DOCUMENT\n"
        "=" * 40 + "\n"
        "Project: Smart City Infrastructure\n"
        "Scope: Design, Build, and Operate a network of IoT sensors.\n"
        "Budget: SAR 15,000,000\n"
        "Timeline: 18 months\n"
        "Evaluation Criteria: Technical (60%), Financial (40%)\n",
        encoding="utf-8",
    )

    # 2. Dummy company profile (also .txt for PoC simplicity)
    company_path = ASSETS_DIR / "company_profile_overview.txt"
    company_path.write_text(
        "COMPANY PROFILE\n"
        "=" * 40 + "\n"
        "Name: Tarseah Technologies\n"
        "Founded: 2020\n"
        "Specialization: AI-driven government solutions\n"
        "Notable Projects: Riyadh Smart Parking, Jeddah Water Monitoring\n",
        encoding="utf-8",
    )

    # 3. Dummy bid details (markdown)
    bid_path = ASSETS_DIR / "bid_details_spec.md"
    bid_path.write_text(
        "# Bid Details\n\n"
        "- **Bid Number**: BID-2026-0042\n"
        "- **Submission Deadline**: 2026-07-15\n"
        "- **Required Certifications**: ISO 27001, ISO 9001\n"
        "- **Mandatory Site Visit**: Yes (Riyadh)\n",
        encoding="utf-8",
    )

    yield  # --- Run the test ---

    # --- Teardown ---
    if ASSETS_DIR.exists():
        shutil.rmtree(ASSETS_DIR)
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestContextInitializerNode:
    """Integration tests for the Context_Initializer_Node."""

    def _run_node(self):
        """Helper to import and execute the node with a minimal state."""
        import sys
        # Ensure src is on the path so relative imports work
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        from nodes.context_initializer import context_initializer_node

        initial_state = {"project_id": TEST_PROJECT_ID}
        return context_initializer_node(initial_state)

    def test_returns_project_id(self):
        """Node must echo back the project_id in its output."""
        result = self._run_node()
        assert result["project_id"] == TEST_PROJECT_ID

    def test_extracts_tender_text(self):
        """Node must extract non-empty tender text."""
        result = self._run_node()
        assert len(result["tender_text"]) > 0
        assert "Smart City Infrastructure" in result["tender_text"]

    def test_extracts_company_text(self):
        """Node must extract non-empty company assets text."""
        result = self._run_node()
        assert len(result["company_assets_text"]) > 0
        assert "Tarseah Technologies" in result["company_assets_text"]

    def test_extracts_bid_text(self):
        """Node must extract non-empty bid details text."""
        result = self._run_node()
        assert len(result["bid_details_text"]) > 0
        assert "BID-2026-0042" in result["bid_details_text"]

    def test_creates_project_storage_directory(self):
        """Node must create the storage/project_{id}/ directory."""
        self._run_node()
        assert STORAGE_DIR.exists()
        assert STORAGE_DIR.is_dir()

    def test_creates_shared_memory_json(self):
        """Node must create shared_memory.json inside the project dir."""
        self._run_node()
        memory_path = STORAGE_DIR / "shared_memory.json"
        assert memory_path.exists()

    def test_shared_memory_has_11_sections(self):
        """shared_memory.json must contain exactly 11 proposal sections."""
        self._run_node()
        memory_path = STORAGE_DIR / "shared_memory.json"
        with open(memory_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        expected_sections = [
            "cover_letter", "executive_summary", "scope_understanding",
            "vision_2030", "company_profile", "past_projects",
            "methodology", "team", "timeline", "quality_and_risk", "pricing",
        ]

        assert len(data["sections"]) == 11
        for section in expected_sections:
            assert section in data["sections"], f"Missing section: {section}"
            assert data["sections"][section]["content"] == ""
            assert data["sections"][section]["status"] == "EMPTY"

    def test_state_contains_shared_memory_path(self):
        """The returned state must include the path to shared_memory.json."""
        result = self._run_node()
        assert "shared_memory_path" in result
        assert Path(result["shared_memory_path"]).exists()

    def test_state_contains_sections_dict(self):
        """The returned state must include an in-memory sections dict."""
        result = self._run_node()
        assert "sections" in result
        assert isinstance(result["sections"], dict)
        assert len(result["sections"]) == 11

    def test_missing_project_id_returns_error(self):
        """If project_id is missing, the node should return an error key."""
        import sys
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        from nodes.context_initializer import context_initializer_node

        result = context_initializer_node({"project_id": ""})
        assert "error" in result
