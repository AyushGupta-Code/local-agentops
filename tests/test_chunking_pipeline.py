"""Unit tests for the chunking and segmentation layer."""

from app.chunking.service import ChunkingService
from app.chunking.utils import approximate_token_count, approximate_word_count
from app.models.schemas import ParsedDocument, ParsedSection


def test_word_and_token_approximation_helpers_return_stable_counts() -> None:
    """Verify that the chunk sizing helpers return predictable approximations."""
    # Use a short normalized sample because the helpers should provide stable
    # counts without depending on a model-specific tokenizer.
    sample_text = "Local AgentOps keeps chunk sizes predictable."

    # Assert that the simple word counter and token estimator produce
    # deterministic, non-zero values for readable text.
    assert approximate_word_count(sample_text) == 6
    assert approximate_token_count(sample_text) >= 6


def test_plain_text_chunking_preserves_overlap_and_deterministic_ids() -> None:
    """Verify that plain-text chunking creates overlapping deterministic windows."""
    # Build a flat parsed document because plain-text chunking should use the
    # sliding-window strategy when no richer structure is available.
    plain_text = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
    )
    document = ParsedDocument(
        document_id="plain-1",
        source_type="document",
        title="Plain Text",
        plain_text=plain_text,
        language="en",
        sections=[],
        source_uri="/tmp/plain.txt",
        metadata={"source_file": "/tmp/plain.txt"},
    )

    # Chunk the document with a small window so multiple overlapping chunks are
    # guaranteed to be produced during the test.
    chunks = ChunkingService(chunk_size=45, chunk_overlap=10).chunk(document)

    # Assert that multiple chunks were created and that their IDs remain
    # deterministic based on document ID and offsets.
    assert len(chunks) >= 2
    assert chunks[0].chunk_id == f"{document.document_id}:0:{chunks[0].char_start}:{chunks[0].char_end}"
    assert chunks[1].char_start < chunks[0].char_end
    assert chunks[1].metadata["source_file"] == "/tmp/plain.txt"

    # Confirm overlap is real by checking that the shared source span appears in
    # both adjacent chunks.
    overlapping_source_text = document.plain_text[chunks[1].char_start : chunks[0].char_end].strip()
    assert overlapping_source_text
    assert overlapping_source_text in chunks[0].text
    assert overlapping_source_text in chunks[1].text


def test_section_based_chunking_preserves_section_titles_and_page_provenance() -> None:
    """Verify that section-aware chunking keeps section labels and page metadata."""
    # Build a parsed document with section structure and a page map because the
    # section-based strategy should preserve both heading and page provenance.
    plain_text = (
        "Overview\nThis section explains the system architecture and goals.\n\n"
        "Details\nThis section goes deeper into indexing and retrieval behavior."
    )
    overview_text = "Overview\nThis section explains the system architecture and goals."
    details_text = "Details\nThis section goes deeper into indexing and retrieval behavior."
    overview_start = 0
    overview_end = len(overview_text)
    details_start = overview_end + 2
    details_end = details_start + len(details_text)
    sections = [
        ParsedSection(
            heading="Overview",
            text=overview_text,
            order=0,
            start_offset=overview_start,
            end_offset=overview_end,
        ),
        ParsedSection(
            heading="Details",
            text=details_text,
            order=1,
            start_offset=details_start,
            end_offset=details_end,
        ),
    ]
    document = ParsedDocument(
        document_id="md-1",
        source_type="document",
        title="Structured Doc",
        plain_text=plain_text,
        language="en",
        sections=sections,
        source_uri="/tmp/structured.md",
        metadata={
            "source_file": "/tmp/structured.md",
            "page_map": [
                {"page_number": 1, "start_offset": 0, "end_offset": 80},
                {"page_number": 2, "start_offset": 80, "end_offset": len(plain_text)},
            ],
        },
    )

    # Chunk the document with a size that forces the second section to split
    # while still preserving section boundaries.
    chunks = ChunkingService(chunk_size=50, chunk_overlap=12).chunk(document)

    # Assert that section titles were preserved and at least one chunk carries
    # page provenance.
    assert chunks[0].section_heading == "Overview"
    assert chunks[0].metadata["section_title"] == "Overview"
    assert chunks[0].metadata["page_numbers"] == [1]
    assert any(chunk.section_heading == "Details" for chunk in chunks)


def test_support_ticket_chunking_preserves_row_range_and_summary_chunk() -> None:
    """Verify that support-ticket chunking carries ticket provenance into every chunk."""
    # Build a parsed support ticket with summary and body sections because the
    # ticket strategy should keep the summary compact and preserve row metadata.
    summary_text = "Title: Login issue\nTicket ID: 9001\nStatus: open\nPriority: high"
    body_text = (
        "Customer cannot log in after a password reset.\n\n"
        "Support asked the customer to clear cookies and retry.\n\n"
        "Customer confirmed the workaround fixed the issue."
    )
    plain_text = f"{summary_text}\n\n{body_text}"
    sections = [
        ParsedSection(
            heading="Ticket Summary",
            text=summary_text,
            order=0,
            start_offset=0,
            end_offset=len(summary_text),
        ),
        ParsedSection(
            heading="Ticket Body",
            text=body_text,
            order=1,
            start_offset=len(summary_text) + 2,
            end_offset=len(plain_text),
        ),
    ]
    document = ParsedDocument(
        document_id="ticket-1",
        source_type="support_ticket",
        title="Login issue",
        plain_text=plain_text,
        language="en",
        sections=sections,
        source_uri="/tmp/tickets.csv",
        metadata={"source_file": "/tmp/tickets.csv", "row_number": 7},
    )

    # Chunk the support ticket with a window size that keeps the summary whole
    # while splitting the conversation body if needed.
    chunks = ChunkingService(chunk_size=90, chunk_overlap=20).chunk(document)

    # Assert that summary and body chunks preserve support-ticket provenance.
    assert chunks[0].section_heading == "Ticket Summary"
    assert chunks[0].metadata["row_range"] == [7, 7]
    assert all(chunk.metadata["source_file"] == "/tmp/tickets.csv" for chunk in chunks)
    assert any(chunk.section_heading == "Ticket Body" for chunk in chunks)


def test_chunk_many_flattens_documents_in_input_order() -> None:
    """Verify that batch chunking preserves document order across the flattened result."""
    # Build two simple parsed documents because the service should flatten their
    # chunks in the same order the documents were supplied.
    first = ParsedDocument(
        document_id="doc-a",
        source_type="document",
        title="Doc A",
        plain_text="one two three four five six seven eight nine ten",
        language="en",
        sections=[],
        metadata={"source_file": "/tmp/a.txt"},
    )
    second = ParsedDocument(
        document_id="doc-b",
        source_type="document",
        title="Doc B",
        plain_text="alpha beta gamma delta epsilon zeta eta theta",
        language="en",
        sections=[],
        metadata={"source_file": "/tmp/b.txt"},
    )

    # Chunk both documents as one batch to exercise the flattening helper.
    chunks = ChunkingService(chunk_size=20, chunk_overlap=5).chunk_many([first, second])

    # Assert that chunks from the first document appear before chunks from the
    # second document in the flattened result.
    first_document_ids = [chunk.document_id for chunk in chunks]
    assert first_document_ids.index("doc-a") < first_document_ids.index("doc-b")
