import io
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


@pytest.fixture
def make_pdf():
    """Factory fixture to create sample in-memory PDFs with arbitrary text."""
    def _create(paragraphs):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        y = 750
        for p in paragraphs:
            c.drawString(72, y, p)
            y -= 30
            if y < 72:
                c.showPage()
                y = 750
        c.save()
        buffer.seek(0)
        return buffer.getvalue()
    return _create


@pytest.fixture
def sample_pitch_pdf(make_pdf):
    """Fixture providing a sample startup pitch deck PDF with Low, Moderate, and High claims."""
    claims = [
        "We achieved $10k MRR with 12% month-on-month growth, verified by Stripe export.",
        "Our platform will capture a massive, trillion-dollar global market with no competitors.",
        "We have significant traction and are rapidly growing in the enterprise sector.",
    ]
    return make_pdf(claims)
