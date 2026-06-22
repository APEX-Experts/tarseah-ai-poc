"""
Text Extraction Utilities
=========================

Provides format-specific text extraction functions for the three input
document types used in proposal generation:

  1. PDF  →  ``extract_text_from_pdf()``   (via ``pypdf``)
  2. DOCX →  ``extract_text_from_docx()``  (via ``python-docx``)
  3. MD / TXT → ``extract_text_from_text_file()`` (plain read)

A convenience dispatcher ``extract_text(file_path)`` auto-detects the format
from the file extension and calls the appropriate function.

Why pypdf over pdfplumber?
--------------------------
  - pypdf is a pure-Python library with zero native dependencies, making it
    ideal for a PoC that needs to run anywhere without complex installs.
  - For production, you may swap in pdfplumber for better table extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Cleaning — Remove PDF/DOCX Extraction Artifacts
# ---------------------------------------------------------------------------

def clean_extracted_text(text: str) -> str:
    """
    Remove common extraction artifacts that waste LLM tokens without
    contributing meaningful content.

    Cleans:
      - Standalone page numbers (e.g. ``\\n  12  \\n``, ``Page 3 of 15``).
      - Repeated header/footer lines (lines appearing 3+ times across the text).
      - Decorative separator lines (underscores, dashes, equals signs).
      - Excessive consecutive blank lines (collapsed to max 2).
      - Trailing/leading whitespace per line.

    Parameters
    ----------
    text : str
        Raw extracted text from a PDF, DOCX, or other document.

    Returns
    -------
    str
        Cleaned text with artifacts removed.
    """
    import re
    from collections import Counter

    if not text:
        return ""

    lines = text.split("\n")

    # ── Pass 1: Remove standalone page numbers ──
    # Matches lines that are just a number, or "Page X of Y", etc.
    page_num_pattern = re.compile(
        r"^\s*(?:(?:Page|الصفحة|صفحة)\s*)?\d+\s*(?:(?:of|من|/)\s*\d+)?\s*$",
        re.IGNORECASE,
    )
    lines = [ln for ln in lines if not page_num_pattern.match(ln)]

    # ── Pass 2: Remove decorative separator lines ──
    # Lines that are just underscores, dashes, equals, dots, or stars
    separator_pattern = re.compile(r"^\s*[_\-=.*─━═▬■□●◆]{3,}\s*$")
    lines = [ln for ln in lines if not separator_pattern.match(ln)]

    # ── Pass 3: Detect and remove repeated header/footer lines ──
    # Lines that appear 3+ times are likely headers/footers repeated per page
    stripped_lines = [ln.strip() for ln in lines]
    line_counts = Counter(stripped_lines)
    # Only remove short lines (< 120 chars) that repeat — long paragraphs may
    # legitimately repeat in templates
    frequent_lines = {
        ln for ln, count in line_counts.items()
        if count >= 3 and 0 < len(ln) < 120
    }
    if frequent_lines:
        logger.debug(
            "Removing %d frequently repeated header/footer patterns.",
            len(frequent_lines),
        )
        lines = [ln for ln in lines if ln.strip() not in frequent_lines]

    # ── Pass 4: Collapse excessive blank lines ──
    cleaned_lines: list[str] = []
    blank_count = 0
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(stripped)

    result = "\n".join(cleaned_lines).strip()
    logger.debug(
        "Text cleaned: %d -> %d chars (%.1f%% reduction).",
        len(text),
        len(result),
        (1 - len(result) / len(text)) * 100 if text else 0,
    )
    return result


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract all text content from a PDF file using ``pypdf``.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to the PDF file.

    Returns
    -------
    str
        Concatenated text from all pages, separated by newlines.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    RuntimeError
        If pypdf fails to read the file (corrupted, encrypted, etc.).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        from pypdf import PdfReader  # lazy import to keep module lightweight

        reader = PdfReader(str(file_path))
        pages_text: list[str] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                pages_text.append(text)
            else:
                logger.warning(
                    "Page %d of '%s' yielded no text (scanned image?).",
                    page_num,
                    file_path.name,
                )

        full_text = "\n".join(pages_text)
        logger.info(
            "Extracted %d characters from %d pages in '%s'.",
            len(full_text),
            len(reader.pages),
            file_path.name,
        )
        return full_text

    except ImportError:
        logger.error(
            "pypdf is not installed. Run: pip install pypdf"
        )
        raise RuntimeError(
            "pypdf library is required for PDF extraction. "
            "Install it with: pip install pypdf"
        )
    except Exception as exc:
        logger.error("Failed to extract text from PDF '%s': %s", file_path, exc)
        raise RuntimeError(f"PDF extraction failed for '{file_path.name}': {exc}") from exc


# ---------------------------------------------------------------------------
# Word / DOCX Extraction
# ---------------------------------------------------------------------------

def extract_text_from_docx(file_path: Path) -> str:
    """
    Extract all paragraph text from a DOCX file using ``python-docx``.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to the .docx file.

    Returns
    -------
    str
        Concatenated paragraph text, separated by newlines.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    try:
        from docx import Document  # lazy import

        doc = Document(str(file_path))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n".join(paragraphs)

        logger.info(
            "Extracted %d characters from %d paragraphs in '%s'.",
            len(full_text),
            len(paragraphs),
            file_path.name,
        )
        return full_text

    except ImportError:
        logger.error(
            "python-docx is not installed. Run: pip install python-docx"
        )
        raise RuntimeError(
            "python-docx library is required for DOCX extraction. "
            "Install it with: pip install python-docx"
        )
    except Exception as exc:
        logger.error("Failed to extract text from DOCX '%s': %s", file_path, exc)
        raise RuntimeError(f"DOCX extraction failed for '{file_path.name}': {exc}") from exc


# ---------------------------------------------------------------------------
# Markdown / Plain-Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_text_file(file_path: Path) -> str:
    """
    Read a plain text or Markdown file and return its content as-is.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to the .md or .txt file.

    Returns
    -------
    str
        Raw file content.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    full_text = file_path.read_text(encoding="utf-8")
    logger.info(
        "Read %d characters from text file '%s'.",
        len(full_text),
        file_path.name,
    )
    return full_text


# ---------------------------------------------------------------------------
# Smart Dispatcher
# ---------------------------------------------------------------------------

# Maps lowercase file extensions to their extraction function.
_EXTRACTOR_MAP = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".doc": extract_text_from_docx,   # .doc may fail; python-docx only supports .docx
    ".md": extract_text_from_text_file,
    ".txt": extract_text_from_text_file,
    ".markdown": extract_text_from_text_file,
    ".json": extract_text_from_text_file,
    ".csv": extract_text_from_text_file,
}


def extract_text(file_path: Path) -> str:
    """
    Auto-detect file format and extract text accordingly.

    Parameters
    ----------
    file_path : Path
        Path to any supported document file.

    Returns
    -------
    str
        Extracted text content.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    extractor = _EXTRACTOR_MAP.get(ext)
    if extractor is None:
        supported = ", ".join(sorted(_EXTRACTOR_MAP.keys()))
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported formats: {supported}"
        )

    raw_text = extractor(file_path)
    return clean_extracted_text(raw_text)


def find_file_by_role(
    directory: Path,
    role: str,
    preferred_extensions: Optional[list[str]] = None,
) -> Optional[Path]:
    """
    Locate a file inside ``directory`` whose name contains the given ``role``
    keyword (case-insensitive).  When multiple matches exist, prefer the file
    whose extension appears first in ``preferred_extensions``.

    This is a convenience helper for the Context_Initializer_Node which needs
    to find the tender, company profile, and bid details files inside an
    arbitrary project directory without hardcoded filenames.

    Parameters
    ----------
    directory : Path
        The project directory to search.
    role : str
        A keyword to match in the filename, e.g. ``"tender"``, ``"company"``,
        ``"bid"``.
    preferred_extensions : list[str] | None
        Ordered list of preferred extensions (e.g. ``[".pdf", ".docx"]``).

    Returns
    -------
    Path | None
        The best matching file path, or ``None`` if no match is found.
    """
    if preferred_extensions is None:
        preferred_extensions = [".pdf", ".docx", ".doc", ".md", ".txt"]

    directory = Path(directory)
    if not directory.exists():
        return None

    candidates: list[Path] = []

    for child in directory.rglob("*"):
        if child.is_file():
            # Check if role matches either the filename or any parent folder name up to the search directory
            relative_path = child.relative_to(directory)
            path_parts = relative_path.parts
            if any(role.lower() in part.lower() for part in path_parts):
                candidates.append(child)

    if not candidates:
        return None

    # Sort by preferred extension order
    def sort_key(p: Path) -> int:
        ext = p.suffix.lower()
        try:
            return preferred_extensions.index(ext)
        except ValueError:
            return len(preferred_extensions)  # unknown ext → lowest priority

    candidates.sort(key=sort_key)
    return candidates[0]
