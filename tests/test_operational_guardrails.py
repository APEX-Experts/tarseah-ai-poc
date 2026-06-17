"""
Tests for Strict Operational Guardrails
=======================================

Verifies:
1. Team Section:
   - Programmatically bypasses LLM and returns the exact fallback statement if no team keywords are present.
   - Post-processes and normalizes LLM outputs that contain missing-info indicators.
2. Timeline:
   - Standardizes timeline using Gregorian calendar only and avoids calendar mixing.
3. Quality and Risk:
   - Sets framing to "Quality Compliance Standards" if certifications do not exist in files.
4. Pricing:
   - Mandates realistic, calculated pricing without blank placeholders, distributed mathematically.
"""

import io
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from main import app
from helpers.shared_memory import load_shared_memory
from nodes.prompts_config import SECTIONS_CONFIG

TEST_PROJECT_ID = "guardrails_test_proj"
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


def test_team_section_missing_info_fallback(monkeypatch):
    """
    Test that if there is no team-related information in the uploaded assets,
    the generator programmatically returns the exact fallback string without calling the LLM.
    """
    client = TestClient(app)

    # 1. Initialize with files that contain absolutely no team keywords
    files = [
        ("file", ("tender_rfp.txt", io.BytesIO(b"Project scope: build roads and bridges. Budget: SAR 10M."), "text/plain")),
        ("file", ("company_profile.txt", io.BytesIO(b"Company founded in 2010. Specialize in infrastructure."), "text/plain")),
    ]
    init_resp = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}", files=files)
    assert init_resp.status_code == 200

    # Mock LLM to verify it is NOT called
    mock_llm_instance = MagicMock()
    import nodes.universal_writer
    monkeypatch.setattr(nodes.universal_writer, "_llm", mock_llm_instance)

    # 2. Call generate section for team
    gen_resp = client.post(f"/proposals/generate/{TEST_PROJECT_ID}/team")
    assert gen_resp.status_code == 200
    body = gen_resp.json()

    # Verify fallback message is returned exactly in Arabic
    assert body["generated_markdown"] == "لا تتوفر معلومات حول فريق العمل في المستندات المقدمة."
    assert mock_llm_instance.invoke.call_count == 0  # Bypassed LLM

    # Verify stored in shared memory
    mem_file = STORAGE_DIR / "shared_memory.json"
    mem_data = json.loads(mem_file.read_text())
    assert mem_data["sections"]["team"]["content"] == "لا تتوفر معلومات حول فريق العمل في المستندات المقدمة."


def test_team_section_post_processing_normalization(monkeypatch):
    """
    Test that if the LLM is called (because keywords exist) but returns a variation
    of missing info, the system normalizes the response to the exact fallback string in Arabic.
    """
    from langchain_core.messages import AIMessage
    client = TestClient(app)

    # Initialize with files that DO contain a keyword like 'team', so programmatic pre-check passes
    files = [
        ("file", ("tender_rfp.txt", io.BytesIO(b"We need a team for smart city project."), "text/plain")),
    ]
    init_resp = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}", files=files)
    assert init_resp.status_code == 200

    # Mock LLM to return an Arabic message stating no info is available
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(
        content="لا يوجد معلومات متوفرة عن فريق العمل.",
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    )
    import nodes.universal_writer
    monkeypatch.setattr(nodes.universal_writer, "_llm", mock_llm_instance)

    # Call generate section
    gen_resp = client.post(f"/proposals/generate/{TEST_PROJECT_ID}/team")
    assert gen_resp.status_code == 200
    body = gen_resp.json()

    # The output should be normalized to the exact guardrail Arabic string
    assert body["generated_markdown"] == "لا تتوفر معلومات حول فريق العمل في المستندات المقدمة."


def test_timeline_system_prompt_guardrails():
    """
    Verify that the system prompt for the timeline section enforces Gregorian calendar
    and bans calendar mixing.
    """
    prompt = SECTIONS_CONFIG["timeline"]["system_prompt"]
    assert "Calendar Standardization" in prompt
    assert "Never mix Hijri and Gregorian calendars" in prompt
    assert "Gregorian Calendar Only" in prompt


def test_quality_and_risk_system_prompt_guardrails():
    """
    Verify that the quality & risk system prompt frames quality strictly as internal
    standards if certifications are missing.
    """
    prompt = SECTIONS_CONFIG["quality_and_risk"]["system_prompt"]
    assert "Quality Certifications Verification" in prompt
    assert "DO NOT claim the company holds them" in prompt
    assert "Quality Compliance Standards" in prompt


def test_pricing_system_prompt_guardrails():
    """
    Verify that the pricing system prompt requires realistic estimation, activity-based costing,
    exact mathematical distribution, team data dependency, and bans placeholders.
    """
    prompt = SECTIONS_CONFIG["pricing"]["system_prompt"]
    assert "Never output blank placeholders" in prompt
    assert "calculate and inject realistic" in prompt
    assert "mathematically distributed" in prompt
    assert "100% mathematical accuracy" in prompt
    assert "Dependence on Team Data" in prompt
    assert "pricing general" in prompt


def test_company_profile_system_prompt_guardrails():
    """
    Verify that the company profile system prompt enforces strict factuality and bans hallucinations.
    """
    prompt = SECTIONS_CONFIG["company_profile"]["system_prompt"]
    assert "ANTI-HALLUCINATION & STRICT FACTUALITY GUARDRAILS" in prompt
    assert "Do NOT add, fabricate, or hallucinate" in prompt
    assert "Do NOT invent any software products" in prompt


def test_past_projects_system_prompt_guardrails():
    """
    Verify that the past projects system prompt enforces strict factuality, prohibits project fabrication,
    and handles missing projects cleanly.
    """
    prompt = SECTIONS_CONFIG["past_projects"]["system_prompt"]
    assert "ANTI-HALLUCINATION GUARDRAILS (NO FABRICATION)" in prompt
    assert "Zero Track Record Hallucinations" in prompt
    assert "You MUST NOT fabricate, invent, or add" in prompt

