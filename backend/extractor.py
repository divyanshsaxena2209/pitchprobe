"""PDF upload validation and layout-aware text extraction."""

import io
import re
from typing import Dict, List

import pdfplumber
from fastapi import HTTPException, UploadFile

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
_WORD_LINE_TOLERANCE = 3.0
_COLUMN_GAP = 36.0
_COLUMN_X_TOLERANCE = 28.0
_MAX_LINE_GAP = 25.0

_METRIC_VALUE = re.compile(
    r"^[₹$€£]?\s*\d[\d,.]*(?:\s?%|\s?[kmb])?$",
    re.IGNORECASE,
)


def _is_short_heading(text: str) -> bool:
    """Identify slide furniture/column labels that should not merge into body text."""
    value = re.sub(r"\s+", " ", text).strip()
    if not value:
        return False
    if re.match(r"^\d{1,2}\s*/\s*[A-Z]", value):
        return True
    if re.match(r"^Q[1-4]\s+\d{4}$", value, re.IGNORECASE):
        return True
    letters = [char for char in value if char.isalpha()]
    all_caps = bool(letters) and all(char.isupper() for char in letters)
    if all_caps and len(value.split()) <= 6 and not re.search(r"[$€£₹%]\s*\d", value):
        return True

    # Short card/column labels such as "Concept + feasibility" or "DVT + closed beta".
    if (
        len(value.split()) <= 5
        and not re.search(r"[.!?;:]$", value)
        and not _METRIC_VALUE.fullmatch(value)
        and not re.search(r"\b(?:is|are|was|were|has|have|had|will|achieved|reached|generated|reduced|increased|grew|expects?|projects?)\b", value, re.IGNORECASE)
    ):
        return True
    return False


def _line_chunks(page) -> List[Dict[str, object]]:
    """Read word coordinates and split each visual row into separate columns/cells."""
    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=_WORD_LINE_TOLERANCE,
        keep_blank_chars=False,
        use_text_flow=False,
    )
    if not words:
        return []

    lines: List[Dict[str, object]] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        matching_line = None
        for line in reversed(lines[-8:]):
            if abs(float(line["top"]) - float(word["top"])) <= _WORD_LINE_TOLERANCE:
                matching_line = line
                break
        if matching_line is None:
            lines.append({
                "top": float(word["top"]),
                "bottom": float(word["bottom"]),
                "words": [word],
            })
        else:
            matching_line["words"].append(word)
            matching_line["top"] = min(float(matching_line["top"]), float(word["top"]))
            matching_line["bottom"] = max(float(matching_line["bottom"]), float(word["bottom"]))

    chunks: List[Dict[str, object]] = []
    for line in lines:
        row_words = sorted(line["words"], key=lambda item: item["x0"])
        groups = []
        current = []
        for word in row_words:
            if current and float(word["x0"]) - float(current[-1]["x1"]) > _COLUMN_GAP:
                groups.append(current)
                current = []
            current.append(word)
        if current:
            groups.append(current)

        for group in groups:
            text = " ".join(str(word["text"]) for word in group)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                chunks.append({
                    "top": float(line["top"]),
                    "bottom": float(line["bottom"]),
                    "x0": float(group[0]["x0"]),
                    "x1": float(group[-1]["x1"]),
                    "text": text,
                })

    return sorted(chunks, key=lambda item: (float(item["top"]), float(item["x0"])))


def _extract_layout_blocks(page) -> List[str]:
    """Rebuild paragraphs by aligned x-position; never join neighbouring slide columns."""
    chunks = _line_chunks(page)
    blocks: List[Dict[str, object]] = []

    for chunk in chunks:
        top = float(chunk["top"])
        x0 = float(chunk["x0"])
        text = str(chunk["text"])
        candidate = None

        # Only continue a nearby paragraph when the x-position aligns and the
        # previous line is not a sentence-ending line or a short heading.
        for block in reversed(blocks):
            line_gap = top - float(block["last_bottom"])
            if line_gap < -3 or line_gap > _MAX_LINE_GAP:
                continue
            if abs(x0 - float(block["x0"])) > _COLUMN_X_TOLERANCE:
                continue
            if (
                _is_short_heading(str(block["last_text"]))
                or _is_short_heading(text)
                or re.search(r"[.!?]\s*$", str(block["last_text"]))
            ):
                break
            candidate = block
            break

        if candidate is None:
            blocks.append({
                "top": top,
                "x0": x0,
                "last_bottom": float(chunk["bottom"]),
                "last_text": text,
                "lines": [text],
            })
        else:
            candidate["lines"].append(text)
            candidate["last_bottom"] = float(chunk["bottom"])
            candidate["last_text"] = text

    ordered = sorted(blocks, key=lambda block: (float(block["top"]), float(block["x0"])))
    entries: List[Dict[str, object]] = []
    for block in ordered:
        text = re.sub(r"\s+", " ", " ".join(block["lines"])).strip()
        if text:
            entries.append({
                "top": float(block["top"]),
                "x0": float(block["x0"]),
                "text": text,
            })

    # Large metric cards often place a value and descriptor on separate lines.
    # Reattach values only to nearby text in the same column.
    merged: List[Dict[str, object]] = []
    for entry in entries:
        text = str(entry["text"])
        match_index = None
        for index in range(len(merged) - 1, max(-1, len(merged) - 10), -1):
            previous = merged[index]
            vertical_gap = float(entry["top"]) - float(previous["top"])
            if vertical_gap > 58:
                continue
            if (
                15 <= vertical_gap <= 58
                and abs(float(entry["x0"]) - float(previous["x0"])) <= _COLUMN_X_TOLERANCE
                and _METRIC_VALUE.fullmatch(str(previous["text"]))
                and len(text.split()) <= 8
            ):
                match_index = index
                break
        if match_index is not None:
            previous = merged[match_index]
            previous["text"] = f'{previous["text"]} {text}'
        else:
            merged.append(dict(entry))
    return [str(item["text"]) for item in merged]


def _normalise_pdf_artifacts(text: str) -> str:
    """Repair common missing-glyph artifacts emitted by embedded slide fonts."""
    value = text or ""
    value = re.sub(r"\(cid:\d+\)", " • ", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<!\w)fi(?!\w)", "→", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<![A-Za-z])n(?=\s?\d)", "₹", value)
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


async def extract_pdf_text(file: UploadFile) -> str:
    """Validate PDF size/type and extract text while preserving slide layout."""
    filename = file.filename or "unknown.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files (.pdf) are supported.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF file is empty (0 bytes).",
        )
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum upload size of {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    try:
        extracted_pages = []
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            if not pdf.pages:
                raise HTTPException(status_code=400, detail="The PDF contains no pages.")

            for page in pdf.pages:
                blocks = _extract_layout_blocks(page)
                if not blocks:
                    # Fallback for PDFs without usable word coordinates.
                    fallback = page.extract_text(x_tolerance=2) or ""
                    blocks = [line.strip() for line in fallback.splitlines() if line.strip()]
                page_text = "\n\n".join(
                    _normalise_pdf_artifacts(block)
                    for block in blocks
                    if _normalise_pdf_artifacts(block)
                )
                if page_text:
                    extracted_pages.append(page_text)
        full_text = "\n\n".join(extracted_pages).strip()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read or parse PDF document: {exc}",
        )

    if not full_text:
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text could be extracted from this PDF. "
                "The document may consist purely of scanned images or non-extractable vectors."
            ),
        )
    return full_text
