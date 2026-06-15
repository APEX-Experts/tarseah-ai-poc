"""
Full Audit Test Suite for the Merged Proposal Initialization API
================================================================

POST /proposals/initialize/{project_id}
  Body: multipart/form-data  →  files: List[UploadFile]

Tests:
  1. Happy path — 3 files → 200, node runs
  2. Upload 1 file only → 200 (partial init is fine)
  3. Upload 5 files → 200 (more than 3 is fine)
  4. Re-upload to existing folder → adds / overwrites, no crash
  5. Missing files field entirely → 422
  6. Invalid extension (.exe) → 400
  7. Invalid MIME → 400
  8. Whitespace project_id → 400
  9. Real PDF extraction
 10. Real DOCX extraction
 11. Corrupted PDF → graceful, no crash
 12. shared_memory.json structure verification
 13. Uploaded files list in response

Run:
    PYTHONPATH=src python -m pytest tests/test_proposal_api_full_audit.py -v
"""

import io
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app
from helpers.config import settings

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def assets_dir(pid: str) -> Path:
    return SRC_DIR / "Assets" / "files" / pid


def storage_dir(pid: str) -> Path:
    return SRC_DIR / "storage" / f"project_{pid}"


TEST_IDS = [
    "audit_happy", "audit_one", "audit_five", "audit_rerun",
    "audit_missing", "audit_badext", "audit_badmime", "audit_badid",
    "audit_pdf", "audit_docx", "audit_corrupt",
]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup():
    for pid in TEST_IDS:
        for d in [assets_dir(pid), storage_dir(pid)]:
            if d.exists():
                shutil.rmtree(d)
    yield
    for pid in TEST_IDS:
        for d in [assets_dir(pid), storage_dir(pid)]:
            if d.exists():
                shutil.rmtree(d)


def _files_payload(*file_tuples):
    """Build a list of (field_name, (filename, content, mime)) for the 'files' field."""
    return [
        ("file", (name, io.BytesIO(content), mime))
        for name, content, mime in file_tuples
    ]


def _standard_3_files():
    """The standard 3 files with recognizable keywords in their names."""
    return _files_payload(
        ("tender_rfp_v2.pdf", b"Fake PDF tender content about smart cities", "application/pdf"),
        ("company_profile.docx", b"Fake DOCX company profile content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("bid_details.md", b"# Bid Details\n- item 1\n- item 2\n", "text/markdown"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. HAPPY PATH — 3 files
# ──────────────────────────────────────────────────────────────────────────────

class TestHappyPath:

    def test_returns_200_success(self, client):
        r = client.post("/proposals/initialize/audit_happy", files=_standard_3_files())
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["project_id"] == "audit_happy"

    def test_uploaded_files_in_response(self, client):
        r = client.post("/proposals/initialize/audit_happy", files=_standard_3_files())
        body = r.json()
        assert len(body["uploaded_files"]) == 3
        names = [f["saved_filename"] for f in body["uploaded_files"]]
        assert "tender_rfp_v2.pdf" in names
        assert "company_profile.docx" in names
        assert "bid_details.md" in names

    def test_node_output_has_all_fields(self, client):
        r = client.post("/proposals/initialize/audit_happy", files=_standard_3_files())
        node = r.json()["node_output"]
        assert "tender_text_length" in node
        assert "company_assets_text_length" in node
        assert "bid_details_text_length" in node
        assert len(node["sections_initialized"]) == 11

    def test_files_saved_on_disk(self, client):
        client.post("/proposals/initialize/audit_happy", files=_standard_3_files())
        ad = assets_dir("audit_happy")
        assert (ad / "tender_rfp_v2.pdf").exists()
        assert (ad / "company_profile.docx").exists()
        assert (ad / "bid_details.md").exists()

    def test_shared_memory_json_created(self, client):
        client.post("/proposals/initialize/audit_happy", files=_standard_3_files())
        mem = storage_dir("audit_happy") / "shared_memory.json"
        assert mem.exists()
        data = json.loads(mem.read_text())
        assert len(data["sections"]) == 11
        for sec in data["sections"].values():
            assert sec["status"] == "EMPTY"
            assert sec["content"] == ""

    def test_bid_md_text_extracted(self, client):
        r = client.post("/proposals/initialize/audit_happy", files=_standard_3_files())
        assert r.json()["node_output"]["bid_details_text_length"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# 2. UPLOAD ONLY 1 FILE
# ──────────────────────────────────────────────────────────────────────────────

class TestSingleFile:

    def test_one_file_still_succeeds(self, client):
        """Even with 1 file, the endpoint should work — node discovers what it can."""
        payload = _files_payload(
            ("bid_details.md", b"# Bid\n- one file only\n", "text/markdown"),
        )
        r = client.post("/proposals/initialize/audit_one", files=payload)
        assert r.status_code == 200
        assert len(r.json()["uploaded_files"]) == 1
        assert r.json()["node_output"]["bid_details_text_length"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# 3. UPLOAD 5 FILES — more than the standard 3
# ──────────────────────────────────────────────────────────────────────────────

class TestManyFiles:

    def test_five_files_accepted(self, client):
        payload = _files_payload(
            ("tender_rfp.pdf", b"tender pdf content", "application/pdf"),
            ("company_profile.docx", b"company docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("bid_details.md", b"# bid md", "text/markdown"),
            ("extra_appendix.txt", b"appendix text", "text/plain"),
            ("past_experience.pdf", b"past exp pdf", "application/pdf"),
        )
        r = client.post("/proposals/initialize/audit_five", files=payload)
        assert r.status_code == 200
        assert len(r.json()["uploaded_files"]) == 5


# ──────────────────────────────────────────────────────────────────────────────
# 4. RE-UPLOAD — folder exists, files are added / overwritten
# ──────────────────────────────────────────────────────────────────────────────

class TestReUpload:

    def test_second_upload_adds_files(self, client):
        """First upload 1 file, then upload 2 more — folder should have all 3."""
        p1 = _files_payload(("tender_rfp.pdf", b"v1", "application/pdf"))
        client.post("/proposals/initialize/audit_rerun", files=p1)

        p2 = _files_payload(
            ("company_profile.docx", b"profile", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("bid_details.md", b"# bid", "text/markdown"),
        )
        r = client.post("/proposals/initialize/audit_rerun", files=p2)
        assert r.status_code == 200

        ad = assets_dir("audit_rerun")
        assert (ad / "tender_rfp.pdf").exists()       # from first call
        assert (ad / "company_profile.docx").exists()  # from second call
        assert (ad / "bid_details.md").exists()        # from second call

    def test_overwrite_same_filename(self, client):
        """Re-uploading the same filename should overwrite it."""
        p1 = _files_payload(("tender_rfp.pdf", b"version-1", "application/pdf"))
        client.post("/proposals/initialize/audit_rerun", files=p1)

        p2 = _files_payload(("tender_rfp.pdf", b"version-2-updated", "application/pdf"))
        client.post("/proposals/initialize/audit_rerun", files=p2)

        content = (assets_dir("audit_rerun") / "tender_rfp.pdf").read_bytes()
        assert content == b"version-2-updated"


# ──────────────────────────────────────────────────────────────────────────────
# 5. NO FILES AT ALL → 422
# ──────────────────────────────────────────────────────────────────────────────

class TestNoFiles:

    def test_no_files_field_returns_422(self, client):
        r = client.post("/proposals/initialize/audit_missing")
        assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# 6. INVALID EXTENSION
# ──────────────────────────────────────────────────────────────────────────────

class TestInvalidExtension:

    def test_exe_rejected(self, client):
        payload = _files_payload(("malware.exe", b"evil", "application/pdf"))
        r = client.post("/proposals/initialize/audit_badext", files=payload)
        assert r.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# 7. INVALID MIME
# ──────────────────────────────────────────────────────────────────────────────

class TestInvalidMime:

    def test_bad_mime_rejected(self, client):
        payload = _files_payload(("tender.pdf", b"x", "application/x-msdownload"))
        r = client.post("/proposals/initialize/audit_badmime", files=payload)
        assert r.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# 8. BAD PROJECT ID
# ──────────────────────────────────────────────────────────────────────────────

class TestBadProjectId:

    def test_whitespace_project_id_rejected(self, client):
        payload = _files_payload(("tender.pdf", b"x", "application/pdf"))
        r = client.post("/proposals/initialize/%20%20", files=payload)
        assert r.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# 9. REAL PDF EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

class TestRealPdf:

    def _make_pdf(self) -> bytes:
        try:
            from pypdf import PdfWriter
            w = PdfWriter()
            w.add_blank_page(72, 72)
            buf = io.BytesIO()
            w.write(buf)
            return buf.getvalue()
        except ImportError:
            pytest.skip("pypdf not installed")

    def test_valid_pdf_no_crash(self, client):
        payload = _files_payload(("tender_rfp.pdf", self._make_pdf(), "application/pdf"))
        r = client.post("/proposals/initialize/audit_pdf", files=payload)
        assert r.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# 10. REAL DOCX EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

class TestRealDocx:

    def _make_docx(self) -> bytes:
        try:
            from docx import Document
            doc = Document()
            doc.add_paragraph("Tarseah Technologies — Company Overview")
            buf = io.BytesIO()
            doc.save(buf)
            return buf.getvalue()
        except ImportError:
            pytest.skip("python-docx not installed")

    def test_docx_text_extracted(self, client):
        payload = _files_payload(
            ("company_profile.docx", self._make_docx(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        )
        r = client.post("/proposals/initialize/audit_docx", files=payload)
        assert r.status_code == 200
        assert r.json()["node_output"]["company_assets_text_length"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# 11. CORRUPTED PDF — should not crash
# ──────────────────────────────────────────────────────────────────────────────

class TestCorruptedPdf:

    def test_corrupt_pdf_does_not_crash(self, client):
        payload = _files_payload(("tender_rfp.pdf", b"NOT A PDF", "application/pdf"))
        r = client.post("/proposals/initialize/audit_corrupt", files=payload)
        assert r.status_code == 200
