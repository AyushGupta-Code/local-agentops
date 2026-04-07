"""Unit tests for the filesystem ingestion pipeline."""

import json
from pathlib import Path

from app.ingestion.service import IngestionService


def test_ingest_directories_discovers_and_registers_supported_document_files(tmp_path: Path) -> None:
    """Verify that text, Markdown, and PDF files are discovered and registered cleanly."""
    # Create a small mix of supported document-style files because the
    # ingestion service should register each one as a raw document.
    (tmp_path / "guide.txt").write_text("Plain text knowledge article.", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Markdown note\n\nUseful details.", encoding="utf-8")
    (tmp_path / "manual.pdf").write_bytes(b"%PDF-1.4\n%stub-pdf-content")

    # Run a full directory ingestion so the test exercises discovery and
    # per-file registration together rather than isolated helper functions.
    report = IngestionService().ingest_directories([tmp_path])

    # Assert that each supported file produced one registered raw document.
    assert len(report.documents) == 3
    assert {document.mime_type for document in report.documents} == {
        "text/plain",
        "text/markdown",
        "application/pdf",
    }

    # Confirm the metadata contract includes the file-provenance fields the
    # later pipeline will depend on.
    for document in report.documents:
        assert document.metadata["file_name"] in {"guide.txt", "notes.md", "manual.pdf"}
        assert "extension" in document.metadata
        assert "source_path" in document.metadata
        assert "file_created_at" in document.metadata
        assert "ingested_at" in document.metadata
        assert document.metadata["source_type"] == "document"

    # Assert that the supported files did not produce ingestion issues.
    assert report.issues == []


def test_ingest_directories_converts_csv_rows_into_support_ticket_records(tmp_path: Path) -> None:
    """Verify that CSV ticket datasets expand into one raw document per row."""
    # Write a small CSV export because ticket datasets commonly arrive as row-
    # based files rather than one file per ticket.
    (tmp_path / "tickets.csv").write_text(
        "ticket_id,subject,description,status,priority,customer_id,tags\n"
        "1001,Login issue,Customer cannot log in,open,high,cust-1,\"billing,urgent\"\n"
        "1002,Reset password,Needs password reset,resolved,normal,cust-2,auth\n",
        encoding="utf-8",
    )

    # Ingest the directory so the service converts each CSV row into a
    # concrete internal raw-document record.
    report = IngestionService().ingest_directories([tmp_path])

    # Assert that the CSV file expanded into two support-ticket documents.
    assert len(report.documents) == 2
    assert all(document.source_type == "support_ticket" for document in report.documents)
    assert [document.external_id for document in report.documents] == ["1001", "1002"]

    # Confirm row-level metadata was attached so later stages can trace records
    # back to the original dataset row.
    assert report.documents[0].metadata["record_index"] == 0
    assert report.documents[1].metadata["record_index"] == 1
    assert report.documents[0].metadata["file_name"] == "tickets.csv"


def test_ingest_directories_converts_json_ticket_arrays_into_internal_records(tmp_path: Path) -> None:
    """Verify that JSON arrays of ticket-like objects become support-ticket records."""
    # Create a JSON array with ticket-shaped objects because the ingestion
    # service should support row-based datasets in JSON form too.
    payload = [
        {
            "ticket_id": "2001",
            "subject": "Sync error",
            "message": "The desktop client cannot sync.",
            "status": "pending",
            "priority": "urgent",
        },
        {
            "ticket_id": "2002",
            "subject": "Billing question",
            "message": "Customer has a duplicate charge.",
            "status": "open",
            "priority": "high",
        },
    ]
    (tmp_path / "support_tickets.json").write_text(json.dumps(payload), encoding="utf-8")

    # Run the directory ingestion so JSON row conversion is exercised through
    # the same public service API used by the rest of the application.
    report = IngestionService().ingest_directories([tmp_path])

    # Assert that both JSON objects were converted into support-ticket records.
    assert len(report.documents) == 2
    assert all(document.source_type == "support_ticket" for document in report.documents)
    assert report.documents[0].title == "Sync error"
    assert report.documents[1].priority == "high"


def test_ingest_directories_records_unsupported_and_corrupt_files_without_stopping_batch(
    tmp_path: Path,
) -> None:
    """Verify that bad files become report issues instead of aborting the batch."""
    # Create one valid text file so the batch still yields a successful
    # document even when other discovered files fail.
    (tmp_path / "article.txt").write_text("Reference content.", encoding="utf-8")

    # Create one unsupported file because discovery should record that it was
    # seen even though no loader exists for it.
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    # Create corrupt supported files so the service can prove it handles
    # malformed input as non-fatal issues.
    (tmp_path / "broken.pdf").write_bytes(b"NOT_A_PDF")
    (tmp_path / "broken.json").write_text("{not-valid-json", encoding="utf-8")

    # Ingest the directory and capture the full batch report.
    report = IngestionService().ingest_directories([tmp_path])

    # Assert that the valid file still produced one document despite the other
    # failures in the same batch.
    assert len(report.documents) == 1
    assert report.documents[0].title == "article"

    # Confirm the batch recorded one issue per bad file rather than raising.
    assert len(report.issues) == 3
    assert {issue.reason for issue in report.issues} == {
        "unsupported_format",
        "source_load_error",
    }
    assert {issue.path.name for issue in report.issues} == {"image.png", "broken.pdf", "broken.json"}


def test_ingest_directories_scans_nested_directories_recursively(tmp_path: Path) -> None:
    """Verify that recursive discovery reaches files inside nested directories."""
    # Create a nested directory tree because recursive scanning is part of the
    # ingestion service contract.
    nested_directory = tmp_path / "nested" / "docs"
    nested_directory.mkdir(parents=True)
    (nested_directory / "kb.md").write_text("Nested knowledge base note.", encoding="utf-8")

    # Ingest the top-level directory so the service must discover the nested
    # Markdown file through recursive scanning.
    report = IngestionService().ingest_directories([tmp_path], recursive=True)

    # Assert that the nested file was discovered and registered successfully.
    assert len(report.documents) == 1
    assert report.documents[0].title == "kb"
