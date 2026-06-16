"""
Universal_Writer_Node — The Second Node in the Proposal Generation Graph
=========================================================================

This node is the core LLM-powered drafting engine. It is invoked
**on-demand** via a FastAPI endpoint whenever the frontend requests a
specific proposal section (e.g. ``methodology``, ``risk_management``).

Execution Flow:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. LOAD INPUTS                                                     │
    │     Extract project_id, current_section, tender_text,              │
    │     and company_assets_text from the LangGraph ProposalState.      │
    │                                                                     │
    │  2. LOAD SHARED MEMORY                                              │
    │     Open storage/project_{project_id}/shared_memory.json.          │
    │     Compile all non-empty sections into a `compiled_memory` block  │
    │     so Gemini can maintain cross-section consistency.               │
    │                                                                     │
    │  3. FETCH SECTION CONFIG                                            │
    │     Look up the target section in SECTIONS_CONFIG to get its       │
    │     unique Arabic system prompt.                                    │
    │                                                                     │
    │  4. CONSTRUCT PROMPT & ENFORCE ARABIC                               │
    │     Build the user prompt injecting tender_text, company_assets,   │
    │     and compiled_memory. Add strict Arabic output instruction.      │
    │                                                                     │
    │  5. CALL GEMINI 1.5 FLASH                                           │
    │     Invoke the model with the system + user prompts.               │
    │                                                                     │
    │  6. PERSIST TO LOCAL JSON                                           │
    │     Write the generated Markdown into shared_memory.json under     │
    │     the section key → this links separate API sessions together.   │
    │                                                                     │
    │  7. RETURN STATE UPDATE                                             │
    │     Return output_markdown + updated sections to LangGraph.        │
    └─────────────────────────────────────────────────────────────────────┘

Why Compiled Memory?
--------------------
Each section is generated in a separate API call. Without shared context,
Gemini would repeat itself or contradict earlier sections. By compiling
all previously approved/drafted sections into the prompt, we give Gemini
a running narrative of the entire proposal — enforcing consistency without
an expensive vector database.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from models.proposal_state import ProposalState
from helpers.shared_memory import (
    load_shared_memory,
    update_section,
    PROPOSAL_SECTIONS,
)
from nodes.prompts_config import get_section_config, SECTIONS_CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Storage root — resolved relative to src/ (same convention as context_initializer)
_STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"

# Groq model configuration
_MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
_TEMPERATURE = 0.4  # Slightly creative but factually grounded
_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# LLM Singleton (module-level to avoid re-initialization on every call)
# ---------------------------------------------------------------------------

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    """
    Lazy-initialize and return the Groq model instance.

    Uses module-level caching so the model is created once and reused
    across all section-generation calls within the same server process.
    """
    global _llm
    if _llm is None:
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY environment variable is not set. ChatGroq may fail.")
        _llm = ChatGroq(
            model=model_name,
            temperature=_TEMPERATURE,
            max_retries=_MAX_RETRIES,
            api_key=groq_api_key,
        )
        logger.info("Groq model initialized: %s (temp=%.2f)", model_name, _TEMPERATURE)
    return _llm


# ---------------------------------------------------------------------------
# Section Arabic Names Mapping
# ---------------------------------------------------------------------------

SECTION_ARABIC_NAMES: Dict[str, str] = {
    "cover_letter": "خطاب التقديم",
    "executive_summary": "الملخص التنفيذي",
    "scope_understanding": "فهم نطاق العمل",
    "vision_2030": "التوافق مع رؤية 2030",
    "company_profile": "ملف الشركة",
    "past_projects": "المشاريع السابقة (سوابق الأعمال)",
    "methodology": "منهجية التنفيذ",
    "team": "هيكل فريق العمل",
    "timeline": "الجدول الزمني لتنفيذ المشروع",
    "quality_and_risk": "إدارة الجودة والمخاطر",
    "pricing": "العرض المالي والتسعير",
}


# ---------------------------------------------------------------------------
# Context Filtering Helpers (Token Reduction)
# ---------------------------------------------------------------------------

SECTION_KEYWORDS: Dict[str, list[str]] = {
    "cover_letter": [
        "خطاب", "تقديم", "رسالة", "تغطية", "عناية", "سعادة", "الموقر", "مقدم", "عرضنا",
        "cover", "letter", "proposal", "tender", "bid", "submission", "gentlemen", "dear"
    ],
    "executive_summary": [
        "ملخص", "تنفيذي", "موجز", "رؤية", "أهداف", "فهم", "قيمة", "مضافة", "تميز",
        "executive", "summary", "overview", "value", "proposition", "goals", "objectives"
    ],
    "scope_understanding": [
        "نطاق", "العمل", "فهم", "متطلبات", "مخرجات", "أهداف", "قيود", "محددات", "كراسة", "الشروط",
        "scope", "understanding", "requirements", "objectives", "deliverables", "constraints", "rfp"
    ],
    "vision_2030": [
        "رؤية", "2030", "مستهدفات", "سعودة", "وطني", "المحتوى", "المحلي", "التحول", "جودة", "الحياة",
        "vision", "alignment", "local content", "saudization", "national", "quality of life"
    ],
    "company_profile": [
        "تأسيس", "رؤية", "رسالة", "قيم", "خبرة", "مجالات", "اعتماد", "شهادات", "أيزو", "iso", "هيكل",
        "profile", "about", "history", "values", "mission", "vision", "organization", "certifications"
    ],
    "past_projects": [
        "سوابق", "أعمال", "مشاريع", "سابقة", "خبرة", "عملاء", "عقد", "تنفيذ", "قيمة",
        "past", "projects", "experience", "track record", "clients", "contracts", "references"
    ],
    "methodology": [
        "منهجية", "طريقة", "تنفيذ", "مراحل", "خطوات", "تسليم", "مخرجات", "deliverable", "agile", "waterfall",
        "methodology", "approach", "phases", "execution", "lifecycle", "stages", "processes"
    ],
    "team": [
        "فريق", "عمل", "هيكل", "إداري", "أدوار", "مسؤوليات", "مدير", "مشروع", "خبرات", "سيرة", "ذاتية",
        "team", "structure", "roles", "responsibilities", "cv", "resume", "staff", "key personnel"
    ],
    "timeline": [
        "جدول", "زمني", "مدة", "أشهر", "أسابيع", "أيام", "معالم", "تسليم", "خطة", "milestones",
        "timeline", "schedule", "duration", "plan", "milestone", "phases", "gantt"
    ],
    "quality_and_risk": [
        "جودة", "مخاطر", "ضمان", "رقابة", "تخفيف", "احتمالية", "أثر", "سجل", "احتياطي", "kpi", "مؤشرات",
        "quality", "risk", "mitigation", "assurance", "control", "likelihood", "impact", "register"
    ],
    "pricing": [
        "مالي", "سعر", "تسعير", "تكلفة", "تكاليف", "جدول", "كميات", "دفع", "شروط", "ضمان",
        "pricing", "financial", "cost", "budget", "payment", "schedule", "commercial", "rates"
    ]
}


def _chunk_text(text: str, chunk_size: int = 1200) -> list[str]:
    """
    Split text into logical chunks of around chunk_size characters,
    preserving paragraph and sentence boundaries where possible.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.؟?!\n])\s+', para)
            for sentence in sentences:
                if current_length + len(sentence) > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [current_chunk[-1]] if len(current_chunk) > 0 else []
                    current_length = sum(len(s) for s in current_chunk)
                current_chunk.append(sentence)
                current_length += len(sentence)
        else:
            if current_length + len(para) > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [current_chunk[-1]]
                current_length = len(current_chunk[0])
            current_chunk.append(para)
            current_length += len(para) + 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def filter_relevant_context(section_key: str, text: str, max_chars: int = 15000) -> str:
    """
    Filters document text to return only the chunks most relevant to the given section.
    If the input text is small (less than max_chars), returns the original text to
    preserve 100% context and ensure backward compatibility/test stability.
    """
    if not text:
        return ""

    import unicodedata
    # Normalize Arabic characters (e.g. presentation forms to standard forms)
    text = unicodedata.normalize("NFKC", text)

    if len(text) <= max_chars:
        return text

    keywords = SECTION_KEYWORDS.get(section_key, [])
    if not keywords:
        return text

    chunks = _chunk_text(text, chunk_size=1200)
    if not chunks:
        return ""

    scored_chunks = []
    for chunk in chunks:
        score = 0
        chunk_lower = chunk.lower()
        for kw in keywords:
            score += chunk_lower.count(kw.lower())
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    top_k = 10
    selected_chunks = []
    for score, chunk in scored_chunks[:top_k]:
        if score > 0:
            selected_chunks.append(chunk)

    if not selected_chunks:
        selected_chunks = [chunk for _, chunk in scored_chunks[:min(3, len(scored_chunks))]]

    return "\n\n...\n\n".join(selected_chunks)


# ---------------------------------------------------------------------------
# Compiled Memory Builder
# ---------------------------------------------------------------------------

def _compile_shared_memory(sections: Dict[str, Dict[str, str]]) -> str:
    """
    Scan all sections in shared_memory and compile non-empty ones into a
    single text block that Gemini can reference for cross-section consistency.

    Parameters
    ----------
    sections : dict
        The ``sections`` dict from shared_memory.json.
        Each key maps to ``{"content": "...", "status": "..."}``.

    Returns
    -------
    str
        A formatted text block containing all previously generated sections.
        Returns an empty string if no sections have content yet.
    """
    memory_parts: list[str] = []

    for section_key in PROPOSAL_SECTIONS:
        entry = sections.get(section_key, {})
        content = entry.get("content", "").strip()

        # Only include sections that have actual generated content
        if content:
            # Use a human-readable label for each section in the compiled block
            readable_label = SECTION_ARABIC_NAMES.get(section_key, section_key.replace("_", " ").title())
            memory_parts.append(
                f"--- قسم: {readable_label} ---\n{content}"
            )

    if not memory_parts:
        return ""

    return (
        "═══════════════════════════════════════════════════\n"
        "الأقسام التي تم إعدادها مسبقاً (للسياق والاتساق):\n"
        "═══════════════════════════════════════════════════\n\n"
        + "\n\n".join(memory_parts)
    )


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def _build_user_prompt(
    section_key: str,
    project_documents_text: str,
    compiled_memory: str,
) -> str:
    """
    Construct the final user prompt that gets sent to Gemini.

    Injects all available context (consolidated project documents, and previously
    generated sections) and enforces strict Arabic Markdown output.

    Parameters
    ----------
    section_key : str
        The target section to generate.
    project_documents_text : str
        Consolidated and filtered text from all project documents.
    compiled_memory : str
        Compiled text of all previously generated sections.

    Returns
    -------
    str
        The complete user prompt ready for Gemini.
    """
    arabic_section_name = SECTION_ARABIC_NAMES.get(section_key, section_key.replace("_", " ").title())

    # -- Core instruction block --
    prompt_parts: list[str] = [
        f"## المطلوب: كتابة قسم «{arabic_section_name}» من العرض الفني\n",
    ]

    # -- Consolidated project documents context --
    if project_documents_text.strip():
        prompt_parts.append(
            "### ١. مستندات ومعلومات المشروع:\n"
            "```\n"
            f"{project_documents_text}\n"
            "```\n"
        )

    # -- Previously generated sections (compiled memory) --
    if compiled_memory.strip():
        prompt_parts.append(
            "### ٢. الأقسام المكتملة مسبقاً (يجب الحفاظ على الاتساق معها وعدم تكرار محتواها):\n"
            f"{compiled_memory}\n"
        )

    # -- Strict output constraints --
    prompt_parts.append(
        "### تعليمات الإخراج الصارمة والملزمة:\n"
        "- **اللغة والترجمة**: يجب أن تكون لغة المخرجات بالكامل هي اللغة العربية الفصحى المهنية الراقية حصرياً.\n"
        "- **الجداول والعناوين**: يجب ترجمة جميع العناوين الرئيسية والفرعية، وأسماء الأعمدة في الجداول، والمدخلات إلى اللغة العربية بالكامل. يمنع منعاً باتاً ترك عناوين الجداول بالإنجليزية.\n"
        "- **التواريخ والمدد**: اكتب جميع التواريخ والمدد الزمنية باللغة العربية حصراً (مثل: 'يناير 2027'، 'مدة 4 أسابيع'، 'سنة واحدة') ولا تستخدم التواريخ بالصيغة الإنجليزية.\n"
        "- **المصطلحات الفنية**: قم بتعريب المصطلحات والمنهجيات الفنية (مثل Agile, Scrum, Gantt, KPIs, Milestones, Stage Gate) وكتابتها باللغة العربية، مع إمكانية ذكر المصطلح الإنجليزي الأصلي بين قوسين فقط عند الضرورة القصوى (مثل: 'منهجية أجايل (Agile)'، 'مؤشرات الأداء الرئيسية (KPIs)'، 'بوابة المرحلة (Stage Gate)').\n"
        "- **التنسيق**: استخدم تنسيق Markdown نظيف ومنظم (عناوين، قوائم، جداول حيثما يناسب).\n"
        "- **ممنوع**: لا تكتب أي مقدمة محادثية أو خاتمة اجتماعية (مثل: 'بالتأكيد'، 'إليك'، 'أتمنى أن يكون مفيداً').\n"
        "- **ممنوع**: لا تكرر محتوى الأقسام المكتملة مسبقاً — أشر إليها فقط عند الحاجة.\n"
        "- **البدء المباشر**: ابدأ مباشرة بمحتوى القسم المطلوب دون أي كلام تمهيدي.\n"
    )

    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# Main Node Function
# ---------------------------------------------------------------------------

def universal_writer_node(state: ProposalState) -> Dict[str, Any]:
    """
    Universal_Writer_Node — LangGraph Node Function.

    Generates a single proposal section using Gemini, persists it to
    the local shared_memory.json file, and returns the updated state.

    Parameters
    ----------
    state : ProposalState
        The current LangGraph state. Must contain:
          - ``project_id``: The project identifier.
          - ``current_section``: The section key to generate.
          - ``tender_text``: Raw tender document text.
          - ``company_assets_text``: Raw company profile text.

    Returns
    -------
    dict
        Partial state update containing:
          - ``output_markdown``: The generated Markdown content.
          - ``sections``: Updated in-memory mirror of all sections.
          - ``current_section``: Echo back the section that was generated.
          - ``error``: Set only if something went wrong.
    """
    # ------------------------------------------------------------------
    # Step 1: Load Inputs from LangGraph State
    # ------------------------------------------------------------------
    project_id_val = state.get("project_id")
    project_id = project_id_val.strip() if isinstance(project_id_val, str) else ""
    
    current_section_val = state.get("current_section")
    current_section = current_section_val.strip() if isinstance(current_section_val, str) else ""
    
    tender_text = state.get("tender_text", "")
    company_assets_text = state.get("company_assets_text", "")
    bid_details_text = state.get("bid_details_text", "")
    additional_assets_text = state.get("additional_assets_text", "")

    # Validate required fields
    if not project_id:
        error_msg = "project_id is required but was not provided in the state."
        logger.error(error_msg)
        return {"error": error_msg}

    if not current_section:
        error_msg = "current_section is required but was not provided in the state."
        logger.error(error_msg)
        return {"error": error_msg}

    # Validate the section key exists in our configuration
    if current_section not in SECTIONS_CONFIG:
        error_msg = (
            f"Unknown section '{current_section}'. "
            f"Valid sections: {list(SECTIONS_CONFIG.keys())}"
        )
        logger.error(error_msg)
        return {"error": error_msg}

    logger.info("═" * 60)
    logger.info("Universal_Writer_Node — START")
    logger.info("  Project: %s | Section: %s", project_id, current_section)
    logger.info("═" * 60)

    # ------------------------------------------------------------------
    # Step 2: Load Shared Memory & Compile Previously Generated Sections
    # ------------------------------------------------------------------
    # The shared_memory.json file is the persistent "brain" that links
    # separate API sessions together. Each call reads what came before
    # and writes its output back — no database needed.
    project_dir = _STORAGE_ROOT / f"project_{project_id}"

    try:
        shared_memory = load_shared_memory(project_dir)
        sections = shared_memory.get("sections", {})
        logger.info(
            "Shared memory loaded. Previously completed sections: %d/%d",
            sum(1 for s in sections.values() if s.get("content", "").strip()),
            len(PROPOSAL_SECTIONS),
        )
    except FileNotFoundError as exc:
        error_msg = (
            f"Shared memory not found for project '{project_id}'. "
            f"Run the Context_Initializer_Node first. Detail: {exc}"
        )
        logger.error(error_msg)
        return {"error": error_msg}

    # Compile all non-empty sections into a text block for Gemini context
    compiled_memory = _compile_shared_memory(sections)
    if compiled_memory:
        logger.info(
            "Compiled memory from %d previous section(s) (%d chars).",
            compiled_memory.count("--- قسم:"),
            len(compiled_memory),
        )
    else:
        logger.info("No previous sections found — this is the first section being generated.")

    # ------------------------------------------------------------------
    # Step 3: Fetch Section Configuration (Arabic system prompt)
    # ------------------------------------------------------------------
    section_config = get_section_config(current_section)
    system_prompt = section_config["system_prompt"]
    section_type = section_config["type"]

    logger.info("Section type: %s | System prompt length: %d chars", section_type, len(system_prompt))

    # ------------------------------------------------------------------
    # Step 4: Construct the Prompt & Enforce Arabic Output
    # ------------------------------------------------------------------
    # Combine all incoming documents into a single consolidated text block
    all_docs = [
        tender_text,
        company_assets_text,
        bid_details_text,
        additional_assets_text
    ]
    combined_docs_text = "\n\n".join(doc for doc in all_docs if doc.strip())

    # Filter the consolidated text block to only include relevant chunks
    filtered_docs = filter_relevant_context(current_section, combined_docs_text)

    logger.info("Context Filtering Stats for section '%s':", current_section)
    logger.info("  Combined Project Documents: %d -> %d chars", len(combined_docs_text), len(filtered_docs))

    user_prompt = _build_user_prompt(
        section_key=current_section,
        project_documents_text=filtered_docs,
        compiled_memory=compiled_memory,
    )

    logger.info("User prompt constructed: %d chars total.", len(user_prompt))

    # ------------------------------------------------------------------
    # Step 5: Call Groq Model
    # ------------------------------------------------------------------
    try:
        llm = _get_llm()

        # Build the message list: system instruction + user prompt
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        logger.info("Invoking Groq for section '%s'...", current_section)
        response = llm.invoke(messages)

        # Extract the generated content from the AIMessage safely
        raw_content = response.content
        if isinstance(raw_content, str):
            generated_markdown = raw_content.strip()
        elif isinstance(raw_content, list):
            parts = []
            for part in raw_content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(str(part["text"]))
            generated_markdown = "".join(parts).strip()
        else:
            generated_markdown = str(raw_content or "").strip()

        logger.info(
            "Groq response received: %d chars for section '%s'.",
            len(generated_markdown),
            current_section,
        )

        # Extract token usage if available
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata:
            input_tokens = usage_metadata.get("input_tokens", 0)
            output_tokens = usage_metadata.get("output_tokens", 0)
        else:
            token_usage = response.response_metadata.get("token_usage", {})
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

        logger.info(
            "Token Usage — Input: %d | Output: %d",
            input_tokens,
            output_tokens,
        )

    except Exception as exc:
        error_msg = f"Groq invocation failed for section '{current_section}': {exc}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

    # ------------------------------------------------------------------
    # Step 6: Persist to Local JSON (shared_memory.json)
    # ------------------------------------------------------------------
    # This is the critical persistence step — it writes the generated
    # content back to the local JSON file so that:
    #   a) The next section call can read it as compiled_memory
    #   b) The frontend can poll the file for progress
    #   c) The data survives server restarts
    try:
        update_section(
            project_dir=project_dir,
            section_key=current_section,
            content=generated_markdown,
            status="DRAFT",
        )
        logger.info(
            "Section '%s' persisted to shared_memory.json with status='DRAFT'.",
            current_section,
        )
    except Exception as exc:
        error_msg = f"Failed to persist section '{current_section}' to shared_memory.json: {exc}"
        logger.error(error_msg, exc_info=True)
        return {
            "output_markdown": generated_markdown,  # Still return the content even if save failed
            "error": error_msg,
        }

    # Reload sections after the update to get the freshest state
    updated_memory = load_shared_memory(project_dir)
    updated_sections = updated_memory.get("sections", {})

    # ------------------------------------------------------------------
    # Step 7: Return State Update to LangGraph
    # ------------------------------------------------------------------
    state_update: Dict[str, Any] = {
        "output_markdown": generated_markdown,
        "sections": updated_sections,
        "current_section": current_section,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    completed_count = sum(
        1 for s in updated_sections.values() if s.get("content", "").strip()
    )

    logger.info("═" * 60)
    logger.info("Universal_Writer_Node — COMPLETE")
    logger.info("  Section: %s | Output: %d chars", current_section, len(generated_markdown))
    logger.info("  Progress: %d/%d sections drafted", completed_count, len(PROPOSAL_SECTIONS))
    logger.info("═" * 60)

    return state_update
