"""Unit tests for the parsing and text extraction layer."""

import csv
import json
from pathlib import Path

import pytest

from app.ingestion.loaders import _build_row_document
from app.models.schemas import RawDocument
from app.parsing.errors import ParsingError
from app.parsing.normalization import normalize_whitespace, remove_broken_text_artifacts
from app.parsing.service import ParsingService


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "parsing"


class FakePdfExtractor:
    """Test-only PDF extractor that returns deterministic page text."""

    def extract_pages(self, path: Path) -> list[str]:
        """Return fixed normalized page content for a PDF fixture path.

        Why this function exists:
            Unit tests should validate parsing behavior without depending on a
            third-party PDF backend or real OCR. This fake extractor lets the
            parsing service exercise page-level provenance deterministically.

        Parameters:
            path: PDF source path supplied by the parsing service.

        Returns:
            A list of per-page text strings used to simulate successful PDF
            extraction.

        Edge cases handled:
            The fake extractor still checks that the fixture exists so tests do
            not accidentally pass with a missing input file.
        """
        # Ensure the test is exercising a real fixture path rather than an
        # accidental nonexistent file.
        assert path.exists()

        # Return one page of text so the parser can build page provenance and
        # section offsets in a deterministic way.
        return ["Hello PDF fixture"]


def test_normalize_whitespace_removes_obvious_artifacts() -> None:
    """Verify that whitespace normalization removes simple extraction noise."""
    # Build a string with null bytes, soft hyphens, redundant spacing, and
    # mixed newlines because those artifacts are common in extracted text.
    raw_text = "Hello\u0000  world\u00ad\r\n\r\n\r\nNext\tline"

    # Normalize the text so the helper contract is exercised directly.
    normalized_text = normalize_whitespace(raw_text)

    # Assert that obvious extraction artifacts and redundant whitespace were
    # removed without destroying paragraph structure.
    assert normalized_text == "Hello world\n\nNext line"
    assert remove_broken_text_artifacts(raw_text).startswith("Hello world")


def test_parse_text_document_fixture_returns_single_section() -> None:
    """Verify that plain-text fixtures become one normalized parsed document."""
    # Load the text fixture into a raw-document wrapper because the parsing
    # service expects ingestion-shaped input.
    fixture_path = FIXTURE_DIRECTORY / "sample.txt"
    raw_document = RawDocument(
        document_id="txt-doc",
        source_type="document",
        connector_name="filesystem",
        title="Sample Text",
        content=fixture_path.read_text(encoding="utf-8"),
        mime_type="text/plain",
        source_uri=str(fixture_path.resolve(strict=False)),
        metadata={"source_path": str(fixture_path.resolve(strict=False))},
    )

    # Parse the fixture through the public parsing service.
    parsed_document = ParsingService().parse(raw_document)

    # Assert that the output is normalized and provenance is preserved.
    assert parsed_document.plain_text.startswith("Local AgentOps is a local-first platform.")
    assert parsed_document.metadata["source_file"].endswith("sample.txt")
    assert len(parsed_document.sections) == 1
    assert parsed_document.sections[0].heading == "Document Body"


def test_parse_markdown_fixture_preserves_headings() -> None:
    """Verify that Markdown fixtures are split into heading-based sections."""
    # Load the Markdown fixture because the parser should preserve its heading
    # structure for later chunking and citations.
    fixture_path = FIXTURE_DIRECTORY / "sample.md"
    raw_document = RawDocument(
        document_id="md-doc",
        source_type="document",
        connector_name="filesystem",
        title="Sample Markdown",
        content=fixture_path.read_text(encoding="utf-8"),
        mime_type="text/markdown",
        source_uri=str(fixture_path.resolve(strict=False)),
        metadata={"source_path": str(fixture_path.resolve(strict=False))},
    )

    # Parse the fixture through the service dispatch.
    parsed_document = ParsingService().parse(raw_document)

    # Assert that the heading structure survived normalization.
    assert [section.heading for section in parsed_document.sections] == ["Overview", "Details"]
    assert "Overview" in parsed_document.plain_text
    assert parsed_document.metadata["section_headings"] == ["Overview", "Details"]


def test_parse_pdf_fixture_preserves_page_provenance() -> None:
    """Verify that PDF parsing preserves page-level provenance metadata."""
    # Build a raw PDF document pointing at the sample fixture while using a
    # fake extractor so the test remains deterministic.
    fixture_path = FIXTURE_DIRECTORY / "sample.pdf"
    raw_document = RawDocument(
        document_id="pdf-doc",
        source_type="document",
        connector_name="filesystem",
        title="Sample PDF",
        content="PDF source registered for later parsing.",
        mime_type="application/pdf",
        source_uri=str(fixture_path.resolve(strict=False)),
        metadata={"source_path": str(fixture_path.resolve(strict=False))},
    )

    # Parse the fixture with the fake extractor to exercise the page-based
    # parsing path without depending on external libraries during tests.
    parsed_document = ParsingService(pdf_extractor=FakePdfExtractor()).parse(raw_document)

    # Assert that page provenance and page heading metadata were preserved.
    assert parsed_document.sections[0].heading == "Page 1"
    assert parsed_document.metadata["page_map"][0]["page_number"] == 1
    assert parsed_document.metadata["source_file"].endswith("sample.pdf")


def test_parse_csv_ticket_fixture_preserves_row_provenance() -> None:
    """Verify that CSV-derived ticket records parse into structured ticket documents."""
    # Load one CSV row from the fixture because the parsing layer receives
    # support tickets after ingestion has already expanded them into raw rows.
    fixture_path = FIXTURE_DIRECTORY / "tickets.csv"
    with fixture_path.open("r", encoding="utf-8", newline="") as file_handle:
        row = next(csv.DictReader(file_handle))

    raw_document = _build_row_document(
        fixture_path,
        row,
        record_index=0,
        connector_name="csv_import",
    )

    # Parse the ticket record through the public service.
    parsed_document = ParsingService().parse(raw_document)

    # Assert that ticket-specific structure and row provenance were preserved.
    assert [section.heading for section in parsed_document.sections] == ["Ticket Summary", "Ticket Body"]
    assert parsed_document.metadata["row_number"] == 0
    assert "Ticket ID: 3001" in parsed_document.plain_text


def test_parse_json_ticket_fixture_preserves_row_provenance() -> None:
    """Verify that JSON-derived ticket records parse into structured ticket documents."""
    # Load the first JSON ticket object from the fixture so the parser can be
    # exercised on the JSON-row support-ticket path.
    fixture_path = FIXTURE_DIRECTORY / "tickets.json"
    row = json.loads(fixture_path.read_text(encoding="utf-8"))[0]
    raw_document = _build_row_document(
        fixture_path,
        row,
        record_index=0,
        connector_name="json_import",
    )

    # Parse the row through the service.
    parsed_document = ParsingService().parse(raw_document)

    # Assert that the JSON ticket retained ticket semantics and provenance.
    assert parsed_document.title == "Billing issue"
    assert parsed_document.metadata["row_number"] == 0
    assert "Priority: urgent" in parsed_document.plain_text


def test_parse_json_document_fixture_formats_json_content() -> None:
    """Verify that document-like JSON content becomes normalized readable text."""
    # Wrap the JSON fixture in a raw-document object because document-style
    # JSON should parse through the non-ticket JSON path.
    fixture_path = FIXTURE_DIRECTORY / "sample.json"
    raw_document = RawDocument(
        document_id="json-doc",
        source_type="document",
        connector_name="json_import",
        title="Sample JSON",
        content=fixture_path.read_text(encoding="utf-8"),
        mime_type="application/json",
        source_uri=str(fixture_path.resolve(strict=False)),
        metadata={"source_path": str(fixture_path.resolve(strict=False))},
    )

    # Parse the fixture using the service dispatch.
    parsed_document = ParsingService().parse(raw_document)

    # Assert that the output retained readable JSON content and provenance.
    assert '"product": "Local AgentOps"' in parsed_document.plain_text
    assert parsed_document.metadata["source_file"].endswith("sample.json")


def test_parsing_service_logs_and_raises_failures(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that parsing failures are logged explicitly before being raised."""
    # Build an unsupported raw document because the service should reject it and
    # log the failure instead of swallowing it.
    raw_document = RawDocument(
        document_id="bad-doc",
        source_type="document",
        connector_name="filesystem",
        title="Bad Document",
        content="data",
        mime_type="application/octet-stream",
        metadata={},
    )

    # Parse the document and assert that the service both raises and logs the
    # unsupported-format failure.
    with pytest.raises(ParsingError):
        ParsingService().parse(raw_document)

    # Confirm the error was logged explicitly, as required by the parsing
    # design contract.
    assert "Unsupported parser input" in caplog.text
