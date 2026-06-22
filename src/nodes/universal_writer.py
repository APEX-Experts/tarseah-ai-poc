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
# Maximum output tokens — MUST be set explicitly.
# Without this, the Groq API applies a small default cap (often 1024-4096)
# which silently truncates large sections (pricing BOQ, timelines, etc.).
# openai/gpt-oss-20b supports up to 65,536 output tokens;
# 16,384 provides ample headroom for the largest Arabic proposal sections.
_MAX_OUTPUT_TOKENS = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "16384"))


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
            max_tokens=_MAX_OUTPUT_TOKENS,
        )
        logger.info(
            "Groq model initialized: %s (temp=%.2f, max_tokens=%d)",
            model_name, _TEMPERATURE, _MAX_OUTPUT_TOKENS,
        )
    return _llm


# ---------------------------------------------------------------------------
# Section Names Mapping (Arabic & English)
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

SECTION_ENGLISH_NAMES: Dict[str, str] = {
    "cover_letter": "Cover Letter",
    "executive_summary": "Executive Summary",
    "scope_understanding": "Understanding of the Scope of Work",
    "vision_2030": "Vision 2030 Alignment",
    "company_profile": "Company Profile",
    "past_projects": "Past Projects (Track Record)",
    "methodology": "Execution Methodology",
    "team": "Project Team and Structure",
    "timeline": "Project Timeline and Schedule",
    "quality_and_risk": "Quality and Risk Management",
    "pricing": "Financial Proposal and Pricing",
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


def has_pricing_info(text: str) -> bool:
    """
    Check if the text contains any pricing/financial cost information.
    Specifically looks for numeric figures associated with currency markers (SAR, USD, ريال, etc.).
    """
    if not text:
        return False
    import re
    import unicodedata
    normalized = unicodedata.normalize("NFKC", text).lower()
    
    # Look for currency keywords
    currency_kw = ["sar", "usd", "ريال", "ر.س", "دولار"]
    has_currency = any(kw in normalized for kw in currency_kw)
    
    # Look for price context words
    price_context_kw = ["سعر", "أسعار", "تكلفة", "تكاليف", "قيمة", "المبلغ", "التسعير", "price", "cost", "pricing", "budget", "rate"]
    has_price_context = any(kw in normalized for kw in price_context_kw)
    
    # Look for numbers (digits greater than 0)
    numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', normalized)
    
    # We consider it has pricing if:
    # 1. It has a currency keyword and any number
    # 2. It has price context keywords and a significant number (e.g. >= 100)
    if has_currency and numbers:
        return True
        
    for num in numbers:
        clean_num = num.replace(',', '').split('.')[0]
        if clean_num.isdigit() and int(clean_num) >= 100:
            if has_price_context:
                return True
                
    return False


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

def _compile_shared_memory(sections: Dict[str, Dict[str, str]], exclude_section: str | None = None) -> str:
    """
    Scan all sections in shared_memory and compile non-empty ones into a
    single text block that Gemini can reference for cross-section consistency.

    Parameters
    ----------
    sections : dict
        The ``sections`` dict from shared_memory.json.
        Each key maps to ``{"content": "...", "status": "..."}``.
    exclude_section : str, optional
        The section key to exclude from the compiled memory block (e.g. the current section being drafted).

    Returns
    -------
    str
        A formatted text block containing all previously generated sections.
        Returns an empty string if no sections have content yet.
    """
    memory_parts: list[str] = []

    for section_key in PROPOSAL_SECTIONS:
        if exclude_section and section_key == exclude_section:
            continue
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
    has_prices: bool = True,
    language: str = "ar",
) -> str:
    """
    Construct the final user prompt that gets sent to the LLM.

    Injects all available context (consolidated project documents, and previously
    generated sections) and enforces strict language-specific Markdown output.

    Parameters
    ----------
    section_key : str
        The target section to generate.
    project_documents_text : str
        Consolidated and filtered text from all project documents.
    compiled_memory : str
        Compiled text of all previously generated sections.
    has_prices : bool, optional
        Flag indicating if the documents contain any pricing information.
    language : str, optional
        Output language — ``'ar'`` (Arabic, default) or ``'en'`` (English).

    Returns
    -------
    str
        The complete user prompt ready for the LLM.
    """
    is_english = language == "en"

    if is_english:
        section_name = SECTION_ENGLISH_NAMES.get(section_key, section_key.replace("_", " ").title())
    else:
        section_name = SECTION_ARABIC_NAMES.get(section_key, section_key.replace("_", " ").title())

    # -- Core instruction block --
    if is_english:
        prompt_parts: list[str] = [
            f"## Required: Write the '{section_name}' section of the Technical Proposal\n",
        ]
    else:
        prompt_parts = [
            f"## المطلوب: كتابة قسم «{section_name}» من العرض الفني\n",
        ]

    # -- Consolidated project documents context --
    if project_documents_text.strip():
        if is_english:
            prompt_parts.append(
                "### 1. Project Documents and Information:\n"
                "```\n"
                f"{project_documents_text}\n"
                "```\n"
            )
        else:
            prompt_parts.append(
                "### ١. مستندات ومعلومات المشروع:\n"
                "```\n"
                f"{project_documents_text}\n"
                "```\n"
            )

    # -- Previously generated sections (compiled memory) --
    if compiled_memory.strip():
        if is_english:
            prompt_parts.append(
                "### 2. Previously Completed Sections (maintain consistency and do not repeat their content):\n"
                f"{compiled_memory}\n"
            )
        else:
            prompt_parts.append(
                "### ٢. الأقسام المكتملة مسبقاً (يجب الحفاظ على الاتساق معها وعدم تكرار محتواها):\n"
                f"{compiled_memory}\n"
            )

    # -- Missing pricing alert --
    if section_key == "pricing" and not has_prices:
        if is_english:
            prompt_parts.append(
                "### CRITICAL ALERT — Missing Pricing Data:\n"
                "- **Very Important Alert**: No pricing information or financial values were found in the provided project documents.\n"
                "- **Mandatory Blank Pricing**: ALL price fields must be left entirely blank (e.g., empty table cells ` | | ` or blank spaces) in all tables and narrative, to be filled manually later. It is absolutely forbidden to guess or invent any numbers or financial estimates for (Unit Price, Total Cost, Payments, Guarantees, Grand Totals).\n"
            )
        else:
            prompt_parts.append(
                "### تنبيه هام جداً بشأن الأسعار المفقودة:\n"
                "- **تنبيه هام جداً**: لم يتم العثور على أي معلومات تسعير أو قيم مالية في مستندات المشروع المقدمة.\n"
                "- **إلزامية ترك الأسعار فارغة**: يجب ترك جميع حقول الأسعار فارغة تماماً (مثل خانة فارغة في الجداول ` | | ` أو مسافة فارغة) في الجداول وفي النص، ليتم تعبئتها يدوياً لاحقاً. يمنع منعاً باتاً تخمين أو اختراع أي أرقام أو تقديرات مالية لـ (السعر الإفرادي، إجمالي التكلفة، الدفعات، الضمانات، المجاميع).\n"
            )

    # -- Strict output constraints --
    if is_english:
        prompt_parts.append(
            "### Strict and Mandatory Output Instructions:\n"
            "- **Language**: The entire output MUST be in professional, high-standard English ONLY.\n"
            "- **Tables and Headings**: All main headings, subheadings, table column names, and entries must be entirely in English.\n"
            "- **Dates and Durations**: Write all dates and durations in English (e.g., 'January 2027', '4 weeks', 'one year').\n"
            "- **Calendar**: Use the Gregorian Calendar ONLY throughout the entire section and document. Do not mix Hijri and Gregorian dates.\n"
            "- **Technical Terms**: Use standard English technical terms. Include original Arabic terms in parentheses only when absolutely necessary for clarity.\n"
            "- **Formatting**: Use clean, structured Markdown format (headings, lists, tables where appropriate).\n"
            "- **Forbidden**: Do not write any conversational preamble or social closing (e.g., 'Sure!', 'Here you go', 'I hope this is helpful').\n"
            "- **Forbidden**: Do not repeat content from previously completed sections — reference them only when needed.\n"
            "- **Direct Start**: Begin directly with the section content without any introductory remarks.\n"
        )
    else:
        prompt_parts.append(
            "### تعليمات الإخراج الصارمة والملزمة:\n"
            "- **اللغة والترجمة**: يجب أن تكون لغة المخرجات بالكامل هي اللغة العربية الفصحى المهنية الراقية حصرياً.\n"
            "- **الجداول والعناوين**: يجب ترجمة جميع العناوين الرئيسية والفرعية، وأسماء الأعمدة في الجداول، والمدخلات إلى اللغة العربية بالكامل. يمنع منعاً باتاً ترك عناوين الجداول بالإنجليزية.\n"
            "- **التواريخ والمدد**: اكتب جميع التواريخ والمدد الزمنية باللغة العربية حصراً (مثل: 'يناير 2027'، 'مدة 4 أسابيع'، 'سنة واحدة') ولا تستخدم التواريخ بالصيغة الإنجليزية.\n"
            "- **التواريخ والتقويم**: يجب استخدام التقويم الميلادي فقط (Gregorian Calendar Only) في كامل القسم والمستند. يمنع منعاً باتاً خلط التواريخ الهجرية والميلادية في نفس القسم أو الجدول (مثال: لا تكتب '1 أكتوبر' متبوعاً بسنة هجرية). يجب أن تكون جميع المدد والمعالم وبوابات المراحل منطقية زمنياً ومتسقة رياضياً.\n"
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
    
    # Language: 'ar' (Arabic) or 'en' (English), default 'ar'
    language = state.get("language", "ar") or "ar"
    if language not in ("ar", "en"):
        language = "ar"
    
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
    logger.info("  Project: %s | Section: %s | Language: %s", project_id, current_section, language)
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

    # Compile all non-empty sections into a text block for Gemini context, excluding the current section
    compiled_memory = _compile_shared_memory(sections, exclude_section=current_section)
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
    section_config = get_section_config(current_section, language=language)
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
    # Filter the consolidated text block to only include relevant chunks
    if current_section == "company_profile":
        # For company profile, use ONLY the company assets text in the input context
        # to ensure the agent gets real data from these files only.
        filtered_docs = company_assets_text
    elif current_section == "past_projects":
        # For past projects, combine the company assets (source of projects) and the tender text
        # (competition RFP details) to allow mapping relevance without hallucination.
        filtered_docs = (
            "=== ملف تعريف الشركة وسوابق الأعمال الحقيقية (المصدر الوحيد لاستخراج المشاريع السابقة) ===\n"
            f"{company_assets_text}\n\n"
            "=== كراسة الشروط والمواصفات للمشروع الحالي (مستندات المنافسة - لتحديد مواءمة سوابق الأعمال فقط) ===\n"
            f"{tender_text}"
        )
    else:
        filtered_docs = filter_relevant_context(current_section, combined_docs_text)

    logger.info("Context Filtering Stats for section '%s':", current_section)
    logger.info("  Combined Project Documents: %d -> %d chars", len(combined_docs_text), len(filtered_docs))

    has_prices = True
    if current_section == "pricing":
        has_prices = has_pricing_info(combined_docs_text)
        logger.info("Pricing check: documents contain pricing information = %s", has_prices)

    user_prompt = _build_user_prompt(
        section_key=current_section,
        project_documents_text=filtered_docs,
        compiled_memory=compiled_memory,
        has_prices=has_prices,
        language=language,
    )

    logger.info("User prompt constructed: %d chars total.", len(user_prompt))

    # Pre-check for team information
    if current_section == "team":
        import re
        import unicodedata
        normalized_docs = unicodedata.normalize("NFKC", combined_docs_text).lower()
        
        # English patterns with word boundaries
        eng_patterns = [
            r"\bteam\b", r"\broles?\b", r"\bpersonnel\b", r"\bstaff\b", r"\bcvs?\b", 
            r"\bresumes?\b", r"\borganogram\b", r"\bstructure\b", r"\bhierarchy\b", 
            r"\bproject manager\b", r"\bqa officer\b", r"\btechnical consultant\b"
        ]
        # Arabic keywords
        ara_keywords = [
            "فريق", "الهيكل التنظيمي", "الهيكل الإداري", "أدوار", "مسؤوليات", 
            "الكوادر", "السير الذاتية", "السيرة الذاتية", "مدير المشروع", "استشاري",
            "مهندس", "مطور", "محلل", "أعضاء"
        ]
        
        has_eng = any(re.search(pat, normalized_docs) for pat in eng_patterns)
        has_ara = any(kw in normalized_docs for kw in ara_keywords)
        has_team_info = has_eng or has_ara
        
        if not has_team_info:
            logger.info("Programmatic check: No team info found in documents. Enforcing guardrail.")
            generated_markdown = (
                "No information about the project team is available in the provided documents."
                if language == "en"
                else "لا تتوفر معلومات حول فريق العمل في المستندات المقدمة."
            )
            
            try:
                update_section(
                    project_dir=project_dir,
                    section_key=current_section,
                    content=generated_markdown,
                    status="DRAFT",
                )
            except Exception as exc:
                logger.error("Failed to persist section '%s': %s", current_section, exc)
                
            updated_memory = load_shared_memory(project_dir)
            updated_sections = updated_memory.get("sections", {})
            return {
                "output_markdown": generated_markdown,
                "sections": updated_sections,
                "current_section": current_section,
                "input_tokens": 0,
                "output_tokens": 0,
            }

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

        # Post-process for team
        if current_section == "team":
            normalized_gen = generated_markdown.lower()
            if (
                "no available information regarding the project team" in normalized_gen
                or "no information about the project team" in normalized_gen
                or "لا تتوفر معلومات" in generated_markdown
                or "لا توجد معلومات" in generated_markdown
                or "لا يوجد معلومات" in generated_markdown
            ):
                generated_markdown = (
                    "No information about the project team is available in the provided documents."
                    if language == "en"
                    else "لا تتوفر معلومات حول فريق العمل في المستندات المقدمة."
                )

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

        # Detect truncation via finish_reason
        finish_reason = None
        gen_info = getattr(response, "response_metadata", {})
        if gen_info:
            finish_reason = gen_info.get("finish_reason")
        was_truncated = finish_reason == "length"
        if was_truncated:
            logger.warning(
                "⚠️ TRUNCATION DETECTED for section '%s'! "
                "finish_reason='length' — the model was forced to stop before completing. "
                "Output tokens used: %d / max: %d. "
                "Consider increasing GROQ_MAX_OUTPUT_TOKENS.",
                current_section, output_tokens, _MAX_OUTPUT_TOKENS,
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
        "was_truncated": was_truncated,
        "finish_reason": finish_reason,
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


# ---------------------------------------------------------------------------
# Streaming Generator (for SSE endpoint)
# ---------------------------------------------------------------------------

async def universal_writer_stream(state: ProposalState):
    """
    Async generator that streams LLM output as Server-Sent Events.

    Reuses the same prompt-building and context-filtering logic as
    ``universal_writer_node`` but calls ``llm.stream()`` instead of
    ``llm.invoke()``, yielding each token chunk as it arrives.

    After streaming completes, the full content is persisted to
    ``shared_memory.json`` and a final ``[DONE]`` event is emitted
    with token usage metadata.

    Yields
    ------
    str
        SSE-formatted lines (``data: ...\\n\\n``).
    """
    import json as _json

    # ------------------------------------------------------------------
    # Steps 1-4: Identical setup to universal_writer_node
    # ------------------------------------------------------------------
    project_id_val = state.get("project_id")
    project_id = project_id_val.strip() if isinstance(project_id_val, str) else ""

    current_section_val = state.get("current_section")
    current_section = current_section_val.strip() if isinstance(current_section_val, str) else ""

    tender_text = state.get("tender_text", "")
    company_assets_text = state.get("company_assets_text", "")
    bid_details_text = state.get("bid_details_text", "")
    additional_assets_text = state.get("additional_assets_text", "")

    # Language: 'ar' (Arabic) or 'en' (English), default 'ar'
    language = state.get("language", "ar") or "ar"
    if language not in ("ar", "en"):
        language = "ar"

    if not project_id:
        yield f"data: {_json.dumps({'error': 'project_id is required'})}\n\n"
        return

    if not current_section:
        yield f"data: {_json.dumps({'error': 'current_section is required'})}\n\n"
        return

    if current_section not in SECTIONS_CONFIG:
        yield f"data: {_json.dumps({'error': f'Unknown section: {current_section}', 'valid_sections': list(SECTIONS_CONFIG.keys())})}\n\n"
        return

    logger.info("═" * 60)
    logger.info("Universal_Writer_Node [STREAM] — START")
    logger.info("  Project: %s | Section: %s | Language: %s", project_id, current_section, language)
    logger.info("═" * 60)

    # Load shared memory
    project_dir = _STORAGE_ROOT / f"project_{project_id}"
    try:
        shared_memory = load_shared_memory(project_dir)
        sections = shared_memory.get("sections", {})
    except FileNotFoundError as exc:
        yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
        return

    compiled_memory = _compile_shared_memory(sections, exclude_section=current_section)

    # Section config
    section_config = get_section_config(current_section, language=language)
    system_prompt = section_config["system_prompt"]

    # Build prompt (same logic as non-streaming)
    all_docs = [tender_text, company_assets_text, bid_details_text, additional_assets_text]
    combined_docs_text = "\n\n".join(doc for doc in all_docs if doc.strip())
    if current_section == "company_profile":
        # For company profile, use ONLY the company assets text in the input context
        # to ensure the agent gets real data from these files only.
        filtered_docs = company_assets_text
    elif current_section == "past_projects":
        # For past projects, combine the company assets (source of projects) and the tender text
        # (competition RFP details) to allow mapping relevance without hallucination.
        filtered_docs = (
            "=== ملف تعريف الشركة وسوابق الأعمال الحقيقية (المصدر الوحيد لاستخراج المشاريع السابقة) ===\n"
            f"{company_assets_text}\n\n"
            "=== كراسة الشروط والمواصفات للمشروع الحالي (مستندات المنافسة - لتحديد مواءمة سوابق الأعمال فقط) ===\n"
            f"{tender_text}"
        )
    else:
        filtered_docs = filter_relevant_context(current_section, combined_docs_text)

    has_prices = True
    if current_section == "pricing":
        has_prices = has_pricing_info(combined_docs_text)
        logger.info("Pricing check [STREAM]: documents contain pricing information = %s", has_prices)

    user_prompt = _build_user_prompt(
        section_key=current_section,
        project_documents_text=filtered_docs,
        compiled_memory=compiled_memory,
        has_prices=has_prices,
        language=language,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Pre-check for team information
    if current_section == "team":
        import re
        import unicodedata
        normalized_docs = unicodedata.normalize("NFKC", combined_docs_text).lower()
        
        # English patterns with word boundaries
        eng_patterns = [
            r"\bteam\b", r"\broles?\b", r"\bpersonnel\b", r"\bstaff\b", r"\bcvs?\b", 
            r"\bresumes?\b", r"\borganogram\b", r"\bstructure\b", r"\bhierarchy\b", 
            r"\bproject manager\b", r"\bqa officer\b", r"\btechnical consultant\b"
        ]
        # Arabic keywords
        ara_keywords = [
            "فريق", "الهيكل التنظيمي", "الهيكل الإداري", "أدوار", "مسؤوليات", 
            "الكوادر", "السير الذاتية", "السيرة الذاتية", "مدير المشروع", "استشاري",
            "مهندس", "مطور", "محلل", "أعضاء"
        ]
        
        has_eng = any(re.search(pat, normalized_docs) for pat in eng_patterns)
        has_ara = any(kw in normalized_docs for kw in ara_keywords)
        has_team_info = has_eng or has_ara
        
        if not has_team_info:
            logger.info("Programmatic check [STREAM]: No team info found in documents. Enforcing guardrail.")
            generated_markdown = (
                "No information about the project team is available in the provided documents."
                if language == "en"
                else "لا تتوفر معلومات حول فريق العمل في المستندات المقدمة."
            )
            
            try:
                update_section(
                    project_dir=project_dir,
                    section_key=current_section,
                    content=generated_markdown,
                    status="DRAFT",
                )
            except Exception as exc:
                logger.error("Failed to persist section '%s': %s", current_section, exc)
                
            yield f"data: {_json.dumps({'chunk': generated_markdown}, ensure_ascii=False)}\n\n"
            
            try:
                updated_memory = load_shared_memory(project_dir)
                updated_sections = updated_memory.get("sections", {})
            except Exception:
                updated_sections = {}
                
            done_payload = {
                "done": True,
                "section_type": current_section,
                "section_config_type": section_config["type"],
                "input_tokens": 0,
                "output_tokens": 0,
                "content_length": len(generated_markdown),
                "sections_progress": {
                    key: {
                        "status": val.get("status", "EMPTY"),
                        "has_content": bool(val.get("content", "").strip()),
                    }
                    for key, val in updated_sections.items()
                },
            }
            yield f"data: {_json.dumps(done_payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

    # ------------------------------------------------------------------
    # Step 5: Stream from Groq
    # ------------------------------------------------------------------
    full_content_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    text_buffer = ""
    finish_reason = None
    was_truncated = False

    try:
        llm = _get_llm()
        logger.info("Streaming Groq for section '%s'...", current_section)

        async for chunk in llm.astream(messages):
            # Extract text content from the streamed AIMessageChunk
            token_text = ""
            if isinstance(chunk.content, str):
                token_text = chunk.content
            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, str):
                        token_text += part
                    elif isinstance(part, dict) and "text" in part:
                        token_text += str(part["text"])

            if token_text:
                full_content_parts.append(token_text)
                text_buffer += token_text

                # Find the last word or sentence boundary character
                last_sep_idx = -1
                for i in range(len(text_buffer) - 1, -1, -1):
                    # Separate on whitespace or common punctuation
                    if text_buffer[i] in (' ', '\n', '\r', '\t', '،', '.', '!', '؟', ':', ';', '-'):
                        last_sep_idx = i
                        break

                # Yield if we found a boundary, or if the buffer is getting too long
                if last_sep_idx != -1:
                    to_yield = text_buffer[:last_sep_idx + 1]
                    text_buffer = text_buffer[last_sep_idx + 1:]
                    yield f"data: {_json.dumps({'chunk': to_yield})}\n\n"
                elif len(text_buffer) > 45:
                    yield f"data: {_json.dumps({'chunk': text_buffer})}\n\n"
                    text_buffer = ""

            # Try to extract token usage from the last chunk's metadata
            usage_metadata = getattr(chunk, "usage_metadata", None)
            if usage_metadata:
                input_tokens = usage_metadata.get("input_tokens", input_tokens)
                output_tokens = usage_metadata.get("output_tokens", output_tokens)
            elif hasattr(chunk, "response_metadata") and chunk.response_metadata:
                token_usage = chunk.response_metadata.get("token_usage", {})
                if token_usage:
                    input_tokens = token_usage.get("prompt_tokens", input_tokens)
                    output_tokens = token_usage.get("completion_tokens", output_tokens)

            # Capture finish_reason from the final chunk (set by Groq on the last streamed chunk)
            chunk_meta = getattr(chunk, "response_metadata", {})
            if chunk_meta and chunk_meta.get("finish_reason"):
                finish_reason = chunk_meta["finish_reason"]
                if finish_reason == "length":
                    was_truncated = True
                    logger.warning(
                        "⚠️ TRUNCATION DETECTED [STREAM] for section '%s'! "
                        "finish_reason='length' — the model was forced to stop before completing. "
                        "Output tokens used: %d / max: %d. "
                        "Consider increasing GROQ_MAX_OUTPUT_TOKENS.",
                        current_section, output_tokens, _MAX_OUTPUT_TOKENS,
                    )

        # Flush any remaining text in the buffer
        if text_buffer:
            yield f"data: {_json.dumps({'chunk': text_buffer})}\n\n"

    except Exception as exc:
        error_msg = f"Groq streaming failed for section '{current_section}': {exc}"
        logger.error(error_msg, exc_info=True)
        yield f"data: {_json.dumps({'error': error_msg})}\n\n"
        return

    finally:
        # ------------------------------------------------------------------
        # Step 6: Persist the full generated content
        # ------------------------------------------------------------------
        generated_markdown = "".join(full_content_parts).strip()
        if current_section == "team" and generated_markdown:
            normalized_gen = generated_markdown.lower()
            if (
                "no available information regarding the project team" in normalized_gen
                or "no information about the project team" in normalized_gen
                or "لا تتوفر معلومات" in generated_markdown
                or "لا توجد معلومات" in generated_markdown
                or "لا يوجد معلومات" in generated_markdown
            ):
                generated_markdown = (
                    "No information about the project team is available in the provided documents."
                    if language == "en"
                    else "لا تتوفر معلومات حول فريق العمل في المستندات المقدمة."
                )

        if generated_markdown:
            logger.info(
                "Persisting generated stream content of %d chars for section '%s' to shared_memory.json",
                len(generated_markdown),
                current_section,
            )
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
                logger.error(
                    "Failed to persist section '%s': %s", current_section, exc, exc_info=True
                )

    # Reload to get fresh progress
    try:
        updated_memory = load_shared_memory(project_dir)
        updated_sections = updated_memory.get("sections", {})
    except Exception:
        updated_sections = {}

    completed_count = sum(
        1 for s in updated_sections.values() if s.get("content", "").strip()
    )

    logger.info("═" * 60)
    logger.info("Universal_Writer_Node [STREAM] — COMPLETE")
    logger.info("  Section: %s | Output: %d chars", current_section, len(generated_markdown))
    logger.info("  Progress: %d/%d sections drafted", completed_count, len(PROPOSAL_SECTIONS))
    logger.info("═" * 60)

    # ------------------------------------------------------------------
    # Step 7: Yield final [DONE] event with metadata
    # ------------------------------------------------------------------
    done_payload = {
        "done": True,
        "section_type": current_section,
        "section_config_type": section_config["type"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "content_length": len(generated_markdown),
        "was_truncated": was_truncated,
        "finish_reason": finish_reason,
        "sections_progress": {
            key: {
                "status": val.get("status", "EMPTY"),
                "has_content": bool(val.get("content", "").strip()),
            }
            for key, val in updated_sections.items()
        },
    }
    if was_truncated:
        done_payload["truncation_warning"] = (
            f"⚠️ Section '{current_section}' was truncated by the model "
            f"(finish_reason='length'). The output may be incomplete. "
            f"Output tokens: {output_tokens}/{_MAX_OUTPUT_TOKENS}."
        )
    yield f"data: {_json.dumps(done_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
