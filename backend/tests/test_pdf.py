import io
import pytest
from fastapi import HTTPException, UploadFile
from backend.extractor import extract_pdf_text, MAX_FILE_SIZE


@pytest.mark.asyncio
async def test_reject_non_pdf():
    file = UploadFile(filename="document.txt", file=io.BytesIO(b"Hello world"))
    with pytest.raises(HTTPException) as exc_info:
        await extract_pdf_text(file)
    assert exc_info.value.status_code == 400
    assert "Only PDF files" in exc_info.value.detail


@pytest.mark.asyncio
async def test_reject_empty_pdf():
    file = UploadFile(filename="empty.pdf", file=io.BytesIO(b""))
    with pytest.raises(HTTPException) as exc_info:
        await extract_pdf_text(file)
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_reject_oversized_pdf():
    # Construct virtual oversized buffer without allocating real RAM
    class DummyOversizedFile(io.BytesIO):
        def __len__(self):
            return MAX_FILE_SIZE + 1024

    dummy_bytes = b"0" * (MAX_FILE_SIZE + 10)
    file = UploadFile(filename="huge.pdf", file=io.BytesIO(dummy_bytes))
    with pytest.raises(HTTPException) as exc_info:
        await extract_pdf_text(file)
    assert exc_info.value.status_code == 400
    assert "exceeds maximum upload size" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_extract_valid_pdf(sample_pitch_pdf):
    file = UploadFile(filename="pitch_deck.pdf", file=io.BytesIO(sample_pitch_pdf))
    text = await extract_pdf_text(file)
    assert len(text) > 0
    assert "MRR" in text
    assert "trillion-dollar" in text
