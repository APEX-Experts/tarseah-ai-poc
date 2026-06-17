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


def test_generate_section_endpoint_success(monkeypatch):
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    
    # We patch the ChatGroq model's invoke method
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(
        content="هذا هو محتوى قسم مقدمة العرض التوضيحي.",
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    )
    
    # Force _get_llm to return our mock
    import nodes.universal_writer
    monkeypatch.setattr(nodes.universal_writer, "_llm", mock_llm_instance)
    
    client = TestClient(app)
 
    # 1. Initialize first
    files = [
        ("file", ("tender_rfp.pdf", io.BytesIO(b"Dummy tender content"), "application/pdf")),
        ("file", ("company_profile.docx", io.BytesIO(b"Dummy company content"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ("file", ("bid_details.md", io.BytesIO(b"# Bid Details"), "text/markdown")),
    ]
    init_resp = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}", files=files)
    assert init_resp.status_code == 200
 
    # 2. Call generate section
    gen_resp = client.post(f"/proposals/generate/{TEST_PROJECT_ID}/cover_letter")
    assert gen_resp.status_code == 200
    body = gen_resp.json()
    assert body["status"] == "success"
    assert body["project_id"] == TEST_PROJECT_ID
    assert body["section_type"] == "cover_letter"
    assert "هذا هو محتوى قسم مقدمة العرض" in body["generated_markdown"]
    assert body["input_tokens"] == 100
    assert body["output_tokens"] == 50
    
    # 3. Check memory file contents
    import json
    mem_file = STORAGE_DIR / "shared_memory.json"
    assert mem_file.exists()
    mem_data = json.loads(mem_file.read_text())
    assert mem_data["sections"]["cover_letter"]["content"] == "هذا هو محتوى قسم مقدمة العرض التوضيحي."
    assert mem_data["sections"]["cover_letter"]["status"] == "DRAFT"


def test_generate_multiple_sections_preserves_content(monkeypatch):
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    
    mock_llm_instance = MagicMock()
    # Return different contents for the two calls
    mock_llm_instance.invoke.side_effect = [
        AIMessage(content="محتوى رسالة التغطية."),
        AIMessage(content="محتوى الملخص التنفيذي.")
    ]
    
    import nodes.universal_writer
    monkeypatch.setattr(nodes.universal_writer, "_llm", mock_llm_instance)
    
    client = TestClient(app)

    # 1. Initialize first
    files = [
        ("file", ("tender_rfp.pdf", io.BytesIO(b"Dummy tender content"), "application/pdf")),
        ("file", ("company_profile.docx", io.BytesIO(b"Dummy company content"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ("file", ("bid_details.md", io.BytesIO(b"# Bid Details"), "text/markdown")),
    ]
    init_resp = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}", files=files)
    assert init_resp.status_code == 200

    # 2. Call generate first section (cover_letter)
    gen_resp1 = client.post(f"/proposals/generate/{TEST_PROJECT_ID}/cover_letter")
    assert gen_resp1.status_code == 200
    
    # 3. Call generate second section (executive_summary)
    gen_resp2 = client.post(f"/proposals/generate/{TEST_PROJECT_ID}/executive_summary")
    assert gen_resp2.status_code == 200
    
    # 4. Check memory file contents - both should be preserved
    import json
    mem_file = STORAGE_DIR / "shared_memory.json"
    assert mem_file.exists()
    mem_data = json.loads(mem_file.read_text())
    
    assert mem_data["sections"]["cover_letter"]["content"] == "محتوى رسالة التغطية."
    assert mem_data["sections"]["cover_letter"]["status"] == "DRAFT"
    assert mem_data["sections"]["executive_summary"]["content"] == "محتوى الملخص التنفيذي."
    assert mem_data["sections"]["executive_summary"]["status"] == "DRAFT"
    
    # 5. Check mock_llm call arguments to verify compiled context inclusion
    # The second call should contain the cover_letter content in the user prompt
    assert mock_llm_instance.invoke.call_count == 2
    second_call_args = mock_llm_instance.invoke.call_args_list[1]
    messages = second_call_args[0][0]
    
    # Locate HumanMessage (typically the second message)
    user_prompt = next(msg.content for msg in messages if msg.__class__.__name__ == "HumanMessage")
    
    # Verify the user prompt contains the compiled memory block with cover_letter content
    assert "محتوى رسالة التغطية." in user_prompt
    assert "الأقسام التي تم إعدادها مسبقاً" in user_prompt


def test_initialize_with_force_reset_clears_content(monkeypatch):
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(content="محتوى رسالة التغطية.")
    
    import nodes.universal_writer
    monkeypatch.setattr(nodes.universal_writer, "_llm", mock_llm_instance)
    
    client = TestClient(app)

    # 1. Initialize first
    files = [
        ("file", ("tender_rfp.pdf", io.BytesIO(b"Dummy tender content"), "application/pdf")),
        ("file", ("company_profile.docx", io.BytesIO(b"Dummy company content"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ("file", ("bid_details.md", io.BytesIO(b"# Bid Details"), "text/markdown")),
    ]
    init_resp = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}", files=files)
    assert init_resp.status_code == 200

    # 2. Call generate section to create a DRAFT section
    gen_resp = client.post(f"/proposals/generate/{TEST_PROJECT_ID}/cover_letter")
    assert gen_resp.status_code == 200
    
    # 3. Initialize again with force_reset=False (default) - should preserve
    init_resp2 = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}", files=files)
    assert init_resp2.status_code == 200
    
    import json
    mem_file = STORAGE_DIR / "shared_memory.json"
    mem_data = json.loads(mem_file.read_text())
    assert mem_data["sections"]["cover_letter"]["content"] == "محتوى رسالة التغطية."
    assert mem_data["sections"]["cover_letter"]["status"] == "DRAFT"
    
    # 4. Initialize again with force_reset=True - should clear everything
    init_resp3 = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}?force_reset=true", files=files)
    assert init_resp3.status_code == 200
    
    mem_data_reset = json.loads(mem_file.read_text())
    assert mem_data_reset["sections"]["cover_letter"]["content"] == ""
    assert mem_data_reset["sections"]["cover_letter"]["status"] == "EMPTY"


def test_generate_section_includes_all_assets_as_context(monkeypatch):
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(content="هذا هو محتوى قسم مقدمة العرض التوضيحي.")
    
    import nodes.universal_writer
    monkeypatch.setattr(nodes.universal_writer, "_llm", mock_llm_instance)
    
    client = TestClient(app)

    # Initialize with primary files + additional MD and TXT files
    files = [
        ("file", ("tender_rfp.txt", io.BytesIO(b"Dummy tender content"), "text/plain")),
        ("file", ("company_profile.txt", io.BytesIO(b"Dummy company content"), "text/plain")),
        ("file", ("bid_details.md", io.BytesIO(b"Custom Bid Details Content"), "text/markdown")),
        ("file", ("extra_specs.md", io.BytesIO(b"Extra MD specifications text"), "text/markdown")),
        ("file", ("project_notes.txt", io.BytesIO(b"Additional TXT notes text"), "text/plain")),
    ]
    init_resp = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}", files=files)
    assert init_resp.status_code == 200

    # Call generate section
    gen_resp = client.post(f"/proposals/generate/{TEST_PROJECT_ID}/cover_letter")
    assert gen_resp.status_code == 200
    
    # Assert that all file contents are in the human prompt sent to the LLM
    assert mock_llm_instance.invoke.call_count == 1
    call_args = mock_llm_instance.invoke.call_args_list[0]
    messages = call_args[0][0]
    user_prompt = next(msg.content for msg in messages if msg.__class__.__name__ == "HumanMessage")

    # Verify primary contexts are present
    assert "Dummy tender content" in user_prompt
    assert "Dummy company content" in user_prompt
    
    # Verify bid_details.md context is present
    assert "Custom Bid Details Content" in user_prompt

    # Verify additional project files (extra_specs.md and project_notes.txt) are present
    assert "Extra MD specifications text" in user_prompt
    assert "Additional TXT notes text" in user_prompt
    assert "مستندات ومعلومات المشروع" in user_prompt


def test_generate_section_stream_endpoint_success(monkeypatch):
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessageChunk
    
    async def mock_astream(*args, **kwargs):
        chunks = ["هذا ", "هو ", "محتوى ", "مقدمة ", "العرض."]
        for i, c in enumerate(chunks):
            if i == len(chunks) - 1:
                yield AIMessageChunk(content=c, usage_metadata={"input_tokens": 80, "output_tokens": 40, "total_tokens": 120})
            else:
                yield AIMessageChunk(content=c)
                
    mock_llm_instance = MagicMock()
    mock_llm_instance.astream = mock_astream
    
    import nodes.universal_writer
    monkeypatch.setattr(nodes.universal_writer, "_llm", mock_llm_instance)
    
    client = TestClient(app)
    
    # 1. Initialize first
    files = [
        ("file", ("tender_rfp.pdf", io.BytesIO(b"Dummy tender content"), "application/pdf")),
        ("file", ("company_profile.docx", io.BytesIO(b"Dummy company content"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ("file", ("bid_details.md", io.BytesIO(b"# Bid Details"), "text/markdown")),
    ]
    init_resp = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}", files=files)
    assert init_resp.status_code == 200
    
    # 2. Call stream endpoint
    response = client.post(f"/proposals/generate/{TEST_PROJECT_ID}/cover_letter/stream")
    assert response.status_code == 200
    assert "data: " in response.text
    
    # 3. Check memory file contents
    import json
    mem_file = STORAGE_DIR / "shared_memory.json"
    assert mem_file.exists()
    mem_data = json.loads(mem_file.read_text())
    assert mem_data["sections"]["cover_letter"]["content"] == "هذا هو محتوى مقدمة العرض."
    assert mem_data["sections"]["cover_letter"]["status"] == "DRAFT"


def test_regenerate_section_excludes_self_from_context(monkeypatch):
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = [
        AIMessage(content="النسخة الأولى من خطاب التقديم."),
        AIMessage(content="النسخة الثانية من خطاب التقديم.")
    ]
    
    import nodes.universal_writer
    monkeypatch.setattr(nodes.universal_writer, "_llm", mock_llm_instance)
    
    client = TestClient(app)
    
    # 1. Initialize
    files = [
        ("file", ("tender_rfp.pdf", io.BytesIO(b"Dummy tender content"), "application/pdf")),
        ("file", ("company_profile.docx", io.BytesIO(b"Dummy company content"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ("file", ("bid_details.md", io.BytesIO(b"# Bid Details"), "text/markdown")),
    ]
    init_resp = client.post(f"/proposals/initialize/{TEST_PROJECT_ID}", files=files)
    assert init_resp.status_code == 200
    
    # 2. Generate first time
    resp1 = client.post(f"/proposals/generate/{TEST_PROJECT_ID}/cover_letter")
    assert resp1.status_code == 200
    
    # 3. Generate second time (regenerate)
    resp2 = client.post(f"/proposals/generate/{TEST_PROJECT_ID}/cover_letter")
    assert resp2.status_code == 200
    
    # 4. Verify that in the second call, "النسخة الأولى" is NOT in the user prompt!
    assert mock_llm_instance.invoke.call_count == 2
    second_call_args = mock_llm_instance.invoke.call_args_list[1]
    messages = second_call_args[0][0]
    user_prompt = next(msg.content for msg in messages if msg.__class__.__name__ == "HumanMessage")
    
    assert "النسخة الأولى" not in user_prompt





