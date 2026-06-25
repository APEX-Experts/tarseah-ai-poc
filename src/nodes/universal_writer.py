"""
Universal_Writer_Node — The Second Node in the Proposal Generation Graph
=========================================================================

This node is the core LLM-powered drafting engine. It is invoked
**on-demand** via a FastAPI endpoint whenever the frontend requests a
specific proposal section (e.g. ``methodology``, ``risk_management``).

Execution Flow:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. LOAD INPUTS                                                     │
    │     Extract project_id, current_section, tender_text,               │
    │     and company_assets_text from the LangGraph ProposalState.       │
    │                                                                     │
    │  2. LOAD SHARED MEMORY                                              │
    │     Open storage/project_{project_id}/shared_memory.json.           │
    │     Compile all non-empty sections into a `compiled_memory` block   │
    │     so the model can maintain cross-section consistency.            │
    │                                                                     │
    │  3. FETCH SECTION CONFIG                                            │
    │     Look up the target section in SECTIONS_CONFIG to get its        │
    │     unique Arabic system prompt.                                    │
    │                                                                     │
    │  4. CONSTRUCT PROMPT & ENFORCE ARABIC                               │
    │     Build the user prompt injecting tender_text, company_assets,    │
    │     and compiled_memory. Add strict Arabic output instruction.      │
    │                                                                     │
    │  5. CALL OPENAI                                                     │
    │     Invoke the model with the system + user prompts.                │
    │                                                                     │
    │  6. PERSIST TO LOCAL JSON                                           │
    │     Write the generated Markdown into shared_memory.json under      │
    │     the section key → this links separate API sessions together.    │
    │                                                                     │
    │  7. RETURN STATE UPDATE                                             │
    │     Return output_markdown + updated sections to LangGraph.         │
    └─────────────────────────────────────────────────────────────────────┘

Why Compiled Memory?
--------------------
Each section is generated in a separate API call. Without shared context,
the model would repeat itself or contradict earlier sections. By compiling
all previously approved/drafted sections into the prompt, we give the model
a running narrative of the entire proposal — enforcing consistency without
an expensive vector database.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
from typing import Any, Dict

from langchain_openai import ChatOpenAI
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

# OpenAI model configuration
_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
_TEMPERATURE = 0.4  # Slightly creative but factually grounded
_MAX_RETRIES = 3
# Maximum output tokens — MUST be set explicitly.
# 16,384 provides ample headroom for the largest Arabic proposal sections
# if using models like gpt-4o or gpt-4-turbo that support up to 16k output tokens.
_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "16384"))


# ---------------------------------------------------------------------------
# LLM Singleton (module-level to avoid re-initialization on every call)
# ---------------------------------------------------------------------------

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """
    Lazy-initialize and return the OpenAI model instance.

    Uses module-level caching so the model is created once and reused
    across all section-generation calls within the same server process.
    """
    global _llm
    if _llm is None:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set. ChatOpenAI may fail.")
        _llm = ChatOpenAI(
            model=model_name,
            temperature=_TEMPERATURE,
            max_retries=_MAX_RETRIES,
            api_key=openai_api_key,
            max_tokens=_MAX_OUTPUT_TOKENS,
        )
        logger.info(
            "OpenAI model initialized: %s (temp=%.2f, max_tokens=%d)",
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
        # Arabic: project scope, requirements, deliverables, constraints, specifications
        "نطاق", "العمل", "فهم", "متطلبات", "مخرجات", "أهداف", "قيود", "محددات", "كراسة", "الشروط",
        "مشروع", "خدمات", "أعمال", "مهام", "أنشطة", "تقنية", "فنية", "وظيفي", "مواصفات", "معايير",
        "اشتراطات", "بنود", "التزام", "تعاقد", "عقد", "نظام", "حل", "تطوير", "تصميم", "تشغيل",
        "صيانة", "دعم", "تدريب", "استشارات", "توريد", "تركيب", "إنشاء", "مقاول", "استلام",
        # English: scope, requirements, deliverables, specifications, compliance
        "scope", "understanding", "requirements", "objectives", "deliverables", "constraints", "rfp",
        "project", "services", "works", "tasks", "activities", "technical", "functional",
        "specifications", "standards", "compliance", "system", "solution", "development",
        "design", "operation", "maintenance", "support", "training", "supply", "installation",
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
        # Arabic: methodology, phases, execution, management, governance
        "منهجية", "طريقة", "تنفيذ", "مراحل", "خطوات", "تسليم", "مخرجات", "إدارة", "تغيير",
        "حوكمة", "رقابة", "متابعة", "تقارير", "أدوات", "سياسات", "إجراءات", "ضبط", "تنسيق",
        "اتصال", "تواصل", "موارد", "انتقال", "اختبار", "مراجعة", "اعتماد", "فحص", "تخطيط",
        "نطاق", "أنشطة", "مهام", "عمليات", "تشغيل", "صيانة", "بوابة",
        # English: methodology, phases, processes, governance, framework
        "methodology", "approach", "phases", "execution", "lifecycle", "stages", "processes",
        "deliverable", "agile", "waterfall", "management", "governance", "monitoring",
        "reporting", "tools", "procedures", "control", "coordination", "communication",
        "resources", "transition", "testing", "review", "approval", "planning",
        "implementation", "framework", "workflow", "strategy", "stage gate",
    ],
    "team": [
        "فريق", "عمل", "هيكل", "إداري", "أدوار", "مسؤوليات", "مدير", "مشروع", "خبرات", "سيرة", "ذاتية",
        "team", "structure", "roles", "responsibilities", "cv", "resume", "staff", "key personnel"
    ],
    "timeline": [
        # Arabic: schedule, duration, milestones, phases, dates, delivery
        "جدول", "زمني", "مدة", "أشهر", "أسابيع", "أيام", "معالم", "تسليم", "خطة",
        "فترة", "بداية", "نهاية", "انتهاء", "إكمال", "إنجاز", "موعد", "تاريخ", "سنة",
        "شهر", "يوم", "أسبوع", "مرحلة", "انطلاق", "بدء", "استلام", "مباشرة", "مراحل",
        "التنفيذ", "العقد", "الضمان", "أنشطة", "مخرجات", "نطاق",
        # English: timeline, schedule, milestones, duration, delivery
        "timeline", "schedule", "duration", "plan", "milestone", "phases", "gantt",
        "milestones", "period", "start", "end", "completion", "deadline", "week",
        "month", "year", "delivery", "handover", "commencement", "contract period",
        "warranty period", "critical path", "activities", "deliverables",
    ],
    "quality_and_risk": [
        "جودة", "مخاطر", "ضمان", "رقابة", "تخفيف", "احتمالية", "أثر", "سجل", "احتياطي", "kpi", "مؤشرات",
        "quality", "risk", "mitigation", "assurance", "control", "likelihood", "impact", "register"
    ],
    "pricing": [
        # Arabic: financial, pricing, costs, quantities, payment, guarantees
        "مالي", "سعر", "تسعير", "تكلفة", "تكاليف", "جدول", "كميات", "دفع", "شروط", "ضمان",
        "بند", "وحدة", "إفرادي", "إجمالي", "مقطوعية", "مناقصة", "عطاء", "ضريبة", "غرامة",
        "خصم", "ريال", "مقابل", "أتعاب", "رسوم", "فاتورة", "مستخلص", "دفعة", "استحقاق",
        "قيمة", "مبلغ", "ميزانية", "أسعار", "عرض", "نطاق", "خدمات", "أعمال", "توريد",
        # English: pricing, financial, BOQ, costs, quantities, payment terms
        "pricing", "financial", "cost", "budget", "payment", "schedule", "commercial", "rates",
        "price", "fee", "amount", "total", "subtotal", "grand total", "boq", "bill of quantities",
        "unit price", "lump sum", "item", "line item", "guarantee", "warranty", "vat", "tax",
        "penalty", "discount", "sar", "bid bond", "performance bond", "advance payment",
        "retention", "invoice", "scope", "deliverables", "services", "supply",
    ]
}


# ---------------------------------------------------------------------------
# Per-Section Document Source Routing
# ---------------------------------------------------------------------------

SECTION_DOC_ROUTING: Dict[str, Dict[str, bool | str]] = {
    "cover_letter":        {"tender": "filter", "company": "filter", "bid": False,    "additional": False},
    "executive_summary":   {"tender": False,    "company": True,     "bid": True,     "additional": False},
    "scope_understanding": {"tender": "filter", "company": False,    "bid": True,     "additional": False},
    "vision_2030":         {"tender": "filter", "company": False,    "bid": False,    "additional": False},
    "company_profile":     {"tender": False,    "company": True,     "bid": False,    "additional": False},
    "past_projects":       {"tender": "filter", "company": True,     "bid": False,    "additional": False},
    "methodology":         {"tender": "filter", "company": False,    "bid": True,     "additional": False},
    "team":                {"tender": False,    "company": True,     "bid": False,    "additional": False},
    "timeline":            {"tender": "filter", "company": False,    "bid": True,     "additional": False},
    "quality_and_risk":    {"tender": "filter", "company": "filter", "bid": False,    "additional": False},
    "pricing":             {"tender": "filter", "company": False,    "bid": True,     "additional": True},
}

SECTION_FILTER_CONFIG: Dict[str, Dict[str, int]] = {
    "scope_understanding": {"max_chars": 60000},  # Needs broadest coverage of RFP scope
    "methodology":         {"max_chars": 45000},  # Needs phase/activity/deliverable details
    "timeline":            {"max_chars": 40000},  # Needs durations/milestones/schedule
    "pricing":             {"max_chars": 55000},  # Needs BOQ/items/quantities/payment terms
}


def _route_documents_for_section(
    section_key: str,
    tender_text: str,
    company_assets_text: str,
    bid_details_text: str,
    additional_assets_text: str,
) -> str:
    """
    Build the consolidated document context for a specific section by
    routing only the relevant document sources.
    """
    if section_key == "past_projects":
        parts = []
        if company_assets_text.strip():
            parts.append(
                "=== ملف تعريف الشركة وسوابق الأعمال الحقيقية "
                "(المصدر الوحيد لاستخراج المشاريع السابقة) ===\n"
                f"{company_assets_text}"
            )
        if tender_text.strip():
            filtered_tender = filter_relevant_context(section_key, tender_text)
            parts.append(
                "=== كراسة الشروط والمواصفات للمشروع الحالي "
                "(مستندات المنافسة - لتحديد مواءمة سوابق الأعمال فقط) ===\n"
                f"{filtered_tender}"
            )
        return "\n\n".join(parts)

    routing = SECTION_DOC_ROUTING.get(section_key)
    if not routing:
        all_text = "\n\n".join(
            doc for doc in [tender_text, company_assets_text, bid_details_text, additional_assets_text]
            if doc.strip()
        )
        return filter_relevant_context(section_key, all_text)

    doc_map = {
        "tender": tender_text,
        "company": company_assets_text,
        "bid": bid_details_text,
        "additional": additional_assets_text,
    }

    parts: list[str] = []
    filter_config = SECTION_FILTER_CONFIG.get(section_key, {})
    section_max_chars = filter_config.get("max_chars", 15000)

    for doc_key, mode in routing.items():
        text = doc_map.get(doc_key, "")
        if not text.strip() or mode is False:
            continue
        if mode == "filter":
            filtered = filter_relevant_context(section_key, text, max_chars=section_max_chars)
            if filtered.strip():
                parts.append(filtered)
        else:  # True — include full text
            parts.append(text)

    return "\n\n".join(parts)


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
        return text[:max_chars]

    chunk_size = 1200
    chunks = _chunk_text(text, chunk_size=chunk_size)
    if not chunks:
        return ""

    # Score each chunk, preserving original index for order restoration
    scored_chunks: list[tuple[int, int, str]] = []
    for idx, chunk in enumerate(chunks):
        score = 0
        chunk_lower = chunk.lower()
        for kw in keywords:
            score += chunk_lower.count(kw.lower())
        scored_chunks.append((score, idx, chunk))

    # Dynamic top_k — scales with the char budget so larger budgets capture more
    top_k = max(10, max_chars // chunk_size)

    # Always include the first 2 chunks (document header/overview is universally relevant)
    guaranteed_count = min(2, len(chunks))
    guaranteed_indices: set[int] = set(range(guaranteed_count))

    selected: list[tuple[int, str]] = []
    total_chars = 0

    # 1. Add guaranteed header chunks first
    for idx in sorted(guaranteed_indices):
        chunk = chunks[idx]
        selected.append((idx, chunk))
        total_chars += len(chunk)

    # 2. Sort by score descending, then add top-k relevant chunks within budget
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    for score, idx, chunk in scored_chunks[:top_k]:
        if idx in guaranteed_indices:
            continue  # Already included
        if score > 0 and total_chars + len(chunk) <= max_chars:
            selected.append((idx, chunk))
            total_chars += len(chunk)

    # 3. Fallback: if nothing matched by keywords, take a few top chunks
    if len(selected) <= guaranteed_count:
        for _, idx, chunk in scored_chunks[:min(5, len(scored_chunks))]:
            if idx not in {s[0] for s in selected} and total_chars + len(chunk) <= max_chars:
                selected.append((idx, chunk))
                total_chars += len(chunk)

    # 4. Re-order by original document position to preserve logical flow
    selected.sort(key=lambda x: x[0])

    logger.info(
        "filter_relevant_context('%s'): %d/%d chunks selected, %d/%d chars (%.0f%% reduction)",
        section_key, len(selected), len(chunks), total_chars, len(text),
        (1 - total_chars / len(text)) * 100 if text else 0,
    )

    return "\n\n...\n\n".join(chunk for _, chunk in selected)


# ---------------------------------------------------------------------------
# Compiled Memory Builder
# ---------------------------------------------------------------------------

_COMPILED_MEMORY_MAX_CHARS = 12000


def _compile_shared_memory(
    sections: Dict[str, Dict[str, str]],
    exclude_section: str | None = None,
    max_total_chars: int = _COMPILED_MEMORY_MAX_CHARS,
) -> str:
    """
    Scan all sections in shared_memory and compile non-empty ones into a
    single text block that the model can reference for cross-section consistency.
    """
    memory_parts: list[str] = []

    header = (
        "═══════════════════════════════════════════════════\n"
        "الأقسام التي تم إعدادها مسبقاً (للسياق والاتساق):\n"
        "═══════════════════════════════════════════════════\n\n"
    )
    budget = max_total_chars - len(header)
    current_size = 0

    for section_key in PROPOSAL_SECTIONS:
        if exclude_section and section_key == exclude_section:
            continue
        entry = sections.get(section_key, {})
        content = entry.get("content", "").strip()
        summary = entry.get("summary", "").strip()

        # Only include sections that have actual generated content
        if content:
            readable_label = SECTION_ARABIC_NAMES.get(section_key, section_key.replace("_", " ").title())
            text_to_inject = summary if summary else content
            label_suffix = " (ملخص)" if summary else ""

            part = f"--- قسم: {readable_label}{label_suffix} ---\n{text_to_inject}"
            part_size = len(part) + 2  # +2 for "\n\n" separator

            if current_size + part_size <= budget:
                memory_parts.append(part)
                current_size += part_size
            else:
                logger.info(
                    "Compiled memory budget reached (%d/%d chars). "
                    "Skipping remaining sections to stay within token limits.",
                    current_size, budget,
                )
                break

    if not memory_parts:
        return ""

    compiled = header + "\n\n".join(memory_parts)
    logger.info(
        "Compiled memory: %d sections included, %d chars total.",
        len(memory_parts), len(compiled),
    )
    return compiled


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

    # -- Certification Guardrails --
    routing = SECTION_DOC_ROUTING.get(section_key, {})
    has_company_profile = routing.get("company", False) is not False

    if not has_company_profile:
        if is_english:
            prompt_parts.append(
                "**CRITICAL ANTI-HALLUCINATION GUARDRAIL: CERTIFICATIONS & CREDENTIALS**\n"
                "- You MUST NOT mention, reference, or imply ANY company certifications, ISO standards, or credentials in your output under any circumstances. Focus entirely on the functional requirements without attaching quality badges.\n"
            )
        else:
            prompt_parts.append(
                "**قيد صارم لمنع التزييف: شهادات الجودة والاعتمادات**\n"
                "- يجب عليك ألا تذكر، أو تشير إلى، أو تلمح لأي شهادات للشركة، أو معايير ISO، أو اعتمادات في مخرجاتك تحت أي ظرف. ركز بالكامل على المتطلبات الوظيفية دون إرفاق شارات أو شهادات جودة.\n"
            )
    else:
        if is_english:
            prompt_parts.append(
                "**CRITICAL ANTI-HALLUCINATION GUARDRAIL: CERTIFICATIONS & CREDENTIALS**\n"
                "- You are STRICTLY FORBIDDEN from inventing, assuming, or injecting any specific quality, technical, or professional certifications (e.g., ISO 9001, ISO 21001, PMP, ITIL) unless they are EXPLICITLY written word-for-word in the provided Company Profile text. If no certifications are listed, you must use generic terminology such as 'Industry Best Practices', 'Internal Quality Frameworks', or 'High Quality Standards'.\n"
            )
        else:
            prompt_parts.append(
                "**قيد صارم لمنع التزييف: شهادات الجودة والاعتمادات**\n"
                "- يمنع منعاً باتاً اختراع، افتراض، أو إضافة أي شهادات جودة أو اعتمادات تقنية أو مهنية (مثل ISO 9001, ISO 21001, PMP, ITIL) ما لم تكن مكتوبة نصاً وبشكل صريح في نص ملف الشركة المرفق. إذا لم تكن هناك شهادات مدرجة، يجب عليك استخدام مصطلحات عامة مثل 'أفضل الممارسات في القطاع' أو 'أطر الجودة الداخلية' أو 'معايير الجودة العالية'.\n"
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

    Generates a single proposal section using OpenAI, persists it to
    the local shared_memory.json file, and returns the updated state.
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

    # Compile all non-empty sections into a text block for LLM context
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
    # Step 4: Construct the Prompt & Enforce Output
    # ------------------------------------------------------------------
    # Route only the relevant documents to this section (token optimization)
    all_docs = [
        tender_text,
        company_assets_text,
        bid_details_text,
        additional_assets_text
    ]
    combined_docs_text = "\n\n".join(doc for doc in all_docs if doc.strip())

    filtered_docs = _route_documents_for_section(
        section_key=current_section,
        tender_text=tender_text,
        company_assets_text=company_assets_text,
        bid_details_text=bid_details_text,
        additional_assets_text=additional_assets_text,
    )

    logger.info("Context Routing Stats for section '%s':", current_section)
    logger.info("  Combined All Documents: %d chars | Routed to section: %d chars (%.0f%% reduction)",
                len(combined_docs_text), len(filtered_docs),
                (1 - len(filtered_docs) / len(combined_docs_text)) * 100 if combined_docs_text else 0)

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
    # Step 5: Call OpenAI Model
    # ------------------------------------------------------------------
    try:
        llm = _get_llm()

        # Build the message list: system instruction + user prompt
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        logger.info("Invoking OpenAI for section '%s'...", current_section)
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
            "OpenAI response received: %d chars for section '%s'.",
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
                "Consider increasing OPENAI_MAX_OUTPUT_TOKENS.",
                current_section, output_tokens, _MAX_OUTPUT_TOKENS,
            )

    except Exception as exc:
        error_msg = f"OpenAI invocation failed for section '{current_section}': {exc}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

    # ------------------------------------------------------------------
    # Step 6: Generate Summary & Persist to Local JSON (shared_memory.json)
    # ------------------------------------------------------------------
    section_summary = ""
    try:
        summary_prompt = (
            f"لخص القسم التالي في 3 إلى 5 نقاط رئيسية (Bullet points) باللغة العربية.\n"
            f"يجب أن يكون الملخص مكثفاً ويحتوي على أهم الحقائق فقط لكي يستخدم كمرجع للأقسام الأخرى.\n\n"
            f"{generated_markdown}"
        )
        summary_messages = [
            SystemMessage(content="أنت مساعد ذكي متخصص في تلخيص المستندات بدقة وبإيجاز شديد."),
            HumanMessage(content=summary_prompt)
        ]
        logger.info("Generating summary for section '%s'...", current_section)
        summary_response = llm.invoke(summary_messages)
        content_val = summary_response.content
        if isinstance(content_val, list):
            content_val = " ".join(str(x) for x in content_val)
        section_summary = content_val.strip()
    except Exception as exc:
        logger.warning("Failed to generate section summary for '%s', continuing without it: %s", current_section, exc)

    try:
        update_section(
            project_dir=project_dir,
            section_key=current_section,
            content=generated_markdown,
            status="DRAFT",
            summary=section_summary,
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
    ``universal_writer_node`` but calls ``llm.astream()`` instead of
    ``llm.invoke()``, yielding each token chunk as it arrives.
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

    # Route only the relevant documents to this section (token optimization)
    all_docs = [tender_text, company_assets_text, bid_details_text, additional_assets_text]
    combined_docs_text = "\n\n".join(doc for doc in all_docs if doc.strip())

    filtered_docs = _route_documents_for_section(
        section_key=current_section,
        tender_text=tender_text,
        company_assets_text=company_assets_text,
        bid_details_text=bid_details_text,
        additional_assets_text=additional_assets_text,
    )

    logger.info("Context Routing Stats [STREAM] for section '%s':", current_section)
    logger.info("  Combined All Documents: %d chars | Routed to section: %d chars (%.0f%% reduction)",
                len(combined_docs_text), len(filtered_docs),
                (1 - len(filtered_docs) / len(combined_docs_text)) * 100 if combined_docs_text else 0)

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
    # Step 5: Stream from OpenAI
    # ------------------------------------------------------------------
    full_content_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    text_buffer = ""
    finish_reason = None
    was_truncated = False

    try:
        llm = _get_llm()
        logger.info("Streaming OpenAI for section '%s'...", current_section)

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

            # Capture finish_reason from the final chunk
            chunk_meta = getattr(chunk, "response_metadata", {})
            if chunk_meta and chunk_meta.get("finish_reason"):
                finish_reason = chunk_meta["finish_reason"]
                if finish_reason == "length":
                    was_truncated = True
                    logger.warning(
                        "⚠️ TRUNCATION DETECTED [STREAM] for section '%s'! "
                        "finish_reason='length' — the model was forced to stop before completing. "
                        "Output tokens used: %d / max: %d. "
                        "Consider increasing OPENAI_MAX_OUTPUT_TOKENS.",
                        current_section, output_tokens, _MAX_OUTPUT_TOKENS,
                    )

        # Flush any remaining text in the buffer
        if text_buffer:
            yield f"data: {_json.dumps({'chunk': text_buffer})}\n\n"

    except Exception as exc:
        error_msg = f"OpenAI streaming failed for section '{current_section}': {exc}"
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
            
            # Generate a summary for compiled memory efficiency
            section_summary = ""
            try:
                llm_summary = _get_llm()
                summary_prompt = (
                    f"لخص القسم التالي في 3 إلى 5 نقاط رئيسية (Bullet points) باللغة العربية.\n"
                    f"يجب أن يكون الملخص مكثفاً ويحتوي على أهم الحقائق فقط لكي يستخدم كمرجع للأقسام الأخرى.\n\n"
                    f"{generated_markdown}"
                )
                summary_messages = [
                    SystemMessage(content="أنت مساعد ذكي متخصص في تلخيص المستندات بدقة وبإيجاز شديد."),
                    HumanMessage(content=summary_prompt)
                ]
                logger.info("Generating summary for streamed section '%s'...", current_section)
                summary_response = llm_summary.invoke(summary_messages)
                content_val = summary_response.content
                if isinstance(content_val, list):
                    content_val = " ".join(str(x) for x in content_val)
                section_summary = content_val.strip()
            except Exception as exc:
                logger.warning("Failed to generate section summary for '%s', continuing without it: %s", current_section, exc)

            try:
                update_section(
                    project_dir=project_dir,
                    section_key=current_section,
                    content=generated_markdown,
                    status="DRAFT",
                    summary=section_summary,
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