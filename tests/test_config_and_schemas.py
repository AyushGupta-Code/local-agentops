"""Focused tests for shared configuration and schema contracts."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_repo_root, resolve_managed_data_path
from app.models.schemas import Citation, GenerationResponse, RawDocument, RetrievalRequest


def test_settings_normalize_data_directories_under_repo_root() -> None:
    """Verify that relative data directories resolve under the repository data root."""
    # Build a settings object using relative paths because that is the most
    # common local-development configuration for this project.
    settings = Settings(
        DATA_ROOT="data",
        RAW_DATA_DIR="raw",
        PROCESSED_DATA_DIR="processed",
        INDEX_DATA_DIR="indexes",
        EVAL_DATA_DIR="eval",
    )

    # Assert that the data root resolves under the repository root so runtime
    # components do not depend on the shell's current working directory.
    assert settings.data_root == get_repo_root() / "data"

    # Assert that each managed subdirectory stays under the shared data root.
    assert settings.raw_data_dir == settings.data_root / "raw"
    assert settings.processed_data_dir == settings.data_root / "processed"
    assert settings.index_data_dir == settings.data_root / "indexes"
    assert settings.eval_data_dir == settings.data_root / "eval"


def test_settings_reject_chunk_overlap_larger_than_chunk_size() -> None:
    """Verify that invalid chunking defaults fail fast during settings validation."""
    # Attempt to build settings with an impossible overlap value because later
    # chunking logic cannot operate correctly when overlap consumes the chunk.
    with pytest.raises(ValidationError):
        Settings(DEFAULT_CHUNK_SIZE=200, DEFAULT_CHUNK_OVERLAP=200)


def test_resolve_managed_data_path_blocks_absolute_and_escape_paths() -> None:
    """Verify that safe path resolution rejects absolute and traversal inputs."""
    # Prepare a trusted base directory because the helper only accepts paths
    # that remain inside this managed location.
    base_directory = (get_repo_root() / "data" / "raw").resolve(strict=False)

    # Assert that an absolute path is rejected before it can be used by
    # storage-oriented code.
    with pytest.raises(ValueError):
        resolve_managed_data_path(base_directory, str(Path("/tmp/outside.txt")))

    # Assert that traversal outside the base directory is rejected as well.
    with pytest.raises(ValueError):
        resolve_managed_data_path(base_directory, "../outside.txt")


def test_raw_document_normalizes_tags_and_accepts_support_ticket_fields() -> None:
    """Verify that raw source records support concrete ticket metadata and tag cleanup."""
    # Build a realistic support-ticket style document because the model should
    # support both documents and support tickets from the start.
    document = RawDocument(
        document_id="ticket-001",
        external_id="ZD-1001",
        source_type="support_ticket",
        connector_name="zendesk",
        title="  Login issue  ",
        content="Customer cannot sign in after password reset.",
        customer_id="cust-7",
        ticket_status="open",
        priority="high",
        tags=[" Billing ", "billing", "urgent"],
    )

    # Assert that text normalization and tag deduplication happened as expected.
    assert document.title == "Login issue"
    assert document.tags == ["billing", "urgent"]


def test_retrieval_request_rejects_blank_query() -> None:
    """Verify that whitespace-only queries are rejected by the retrieval contract."""
    # Attempt to build a retrieval request with a blank query because later
    # retrieval stages require meaningful query text.
    with pytest.raises(ValidationError):
        RetrievalRequest(query="   ")


def test_generation_response_keeps_concrete_citations() -> None:
    """Verify that generated answers carry structured citation objects instead of plain strings."""
    # Build a realistic citation because the generation response should carry
    # enough metadata for UI rendering and later evaluation.
    citation = Citation(
        chunk_id="doc-1:0",
        document_id="doc-1",
        source_type="document",
        title="Password Reset Guide",
        locator="Section: Reset Steps",
        snippet="Use the password reset link from the login page.",
    )

    # Create a generation response that references the citation so the model
    # contract reflects future grounded-answer behavior.
    response = GenerationResponse(
        answer="Use the reset link from the login page.",
        citations=[citation],
        used_chunk_ids=["doc-1:0"],
        model_name="llama3.1",
    )

    # Assert that the response keeps the structured citation object intact.
    assert response.citations[0].title == "Password Reset Guide"
