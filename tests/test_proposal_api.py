"""
Basic smoke test for the merged Proposal Initialization API.

Run:
    PYTHONPATH=src python -m pytest tests/test_proposal_api.py -v
"""

import io
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app

TEST_PROJECT_ID = "api_test_proj_999"
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
ASSETS_DIR = SRC_DIR / "Assets" / "files" / TEST_PROJECT_ID
STORAGE_DIR = SRC_DIR / "storage" / f"project_{TEST_PROJECT_ID}"


@pytest.fixture(autouse=True)
def clean_directories():
    for d in [ASSETS_DIR, STORAGE_DIR]:
        if d.exists():
            shutil.rmtree(d)
    yield
    for d in [ASSETS_DIR, STORAGE_DIR]:
        if d.exists():
            shutil.rmtree(d)


def test_initialize_proposal_endpoint_success():
    client = TestClient(app)

    files = [
        ("file", ("tender_rfp.pdf", io.BytesIO(b"Dummy tender content"), "application/pdf")),
        ("file", ("company_profile.docx", io.BytesIO(b"Dummy company content"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ("file", ("bid_details.md", io.BytesIO(b"# Bid Details"), "text/markdown")),
    ]

    response = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["project_id"] == TEST_PROJECT_ID
    assert "shared_memory_path" in body
    assert len(body["uploaded_files"]) == 3
    assert len(body["node_output"]["sections_initialized"]) == 11

    assert (ASSETS_DIR / "tender_rfp.pdf").exists()
    assert (ASSETS_DIR / "company_profile.docx").exists()
    assert (ASSETS_DIR / "bid_details.md").exists()
    assert (STORAGE_DIR / "shared_memory.json").exists()
