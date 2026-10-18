import io
import pytest
from fastapi import HTTPException, UploadFile
from backend.extractor import (
    extract_pdf_text,
    MAX_FILE_SIZE,
    _extract_layout_blocks,
    _normalise_pdf_artifacts,
)


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



def test_layout_extractor_keeps_two_slide_columns_separate():
    class FakePage:
        def extract_words(self, **_kwargs):
            rows = [
                (
                    100.0,
                    [
                        (50.0, "Our"), (75.0, "platform"), (125.0, "reduces"),
                        (175.0, "invoice"), (220.0, "processing"), (285.0, "time"),
                        (315.0, "by"), (335.0, "70%"),
                    ],
                ),
                (
                    100.0,
                    [
                        (600.0, "Revenue"), (650.0, "reached"), (700.0, "10"),
                        (720.0, "crore"), (765.0, "this"), (795.0, "fiscal"),
                        (840.0, "year"),
                    ],
                ),
                (
                    115.0,
                    [
                        (50.0, "for"), (75.0, "small"), (110.0, "business"),
                        (165.0, "customers"), (235.0, "using"), (280.0, "paid"),
                        (315.0, "invoices"),
                    ],
                ),
                (
                    115.0,
                    [
                        (600.0, "from"), (635.0, "hardware"), (700.0, "sales"),
                        (740.0, "and"), (765.0, "paid"), (800.0, "plans"),
                    ],
                ),
            ]
            words = []
            for top, row in rows:
                for x0, word in row:
                    words.append({
                        "text": word,
                        "x0": x0,
                        "x1": x0 + max(8.0, len(word) * 6.0),
                        "top": top,
                        "bottom": top + 10.0,
                    })
            return words

    blocks = _extract_layout_blocks(FakePage())
    assert len(blocks) == 2
    assert blocks[0] == (
        "Our platform reduces invoice processing time by 70% "
        "for small business customers using paid invoices"
    )
    assert blocks[1] == (
        "Revenue reached 10 crore this fiscal year "
        "from hardware sales and paid plans"
    )


def test_pdf_artifact_normalization_repairs_known_font_glyphs():
    normalized = _normalise_pdf_artifacts(
        "Sonora (cid:127) summary fi task n17,999"
    )
    assert normalized == "Sonora • summary → task ₹17,999"
