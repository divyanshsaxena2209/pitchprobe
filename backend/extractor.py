import io
from fastapi import HTTPException, UploadFile
import pdfplumber

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


async def extract_pdf_text(file: UploadFile) -> str:
    """Validate uploaded PDF, enforce 20MB limit, and extract text without permanent disk storage."""
    # 1. Filename validation
    filename = file.filename or "unknown.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files (.pdf) are supported.",
        )

    # 2. In-memory read and size validation
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF file is empty (0 bytes).",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum upload size of {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    # 3. PDF parsing with pdfplumber in-memory (no disk persistence)
    try:
        pdf_stream = io.BytesIO(contents)
        extracted_pages = []
        with pdfplumber.open(pdf_stream) as pdf:
            if not pdf.pages:
                raise HTTPException(
                    status_code=400,
                    detail="The PDF contains no pages.",
                )
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text.strip())

        full_text = "\n\n".join(extracted_pages).strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read or parse PDF document: {str(e)}",
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
