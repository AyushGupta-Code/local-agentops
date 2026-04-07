"""Unit tests for vector and hybrid retrieval execution."""

from pathlib import Path
from shutil import copyfile

from app.core.config import Settings
from app.models.schemas import RetrievalRequest
from app.retrieval.embeddings import HashEmbeddingService
from app.retrieval.service import RetrievalService


def _build_settings(tmp_path: Path) -> Settings:
    """Create test settings rooted in a temporary directory.

    Why this function exists:
        Retrieval tests need isolated storage paths so fixture manifests can be
        copied into a predictable local index directory without touching
        repository data.

    Parameters:
        tmp_path: Pytest-provided temporary directory used as the managed data
            root.

    Returns:
        A ``Settings`` instance whose managed directories live under
        ``tmp_path``.

    Edge cases handled:
        The helper always uses explicit relative subdirectory names so the
        managed path validation in settings is exercised as part of the test.
    """
    return Settings(
        DATA_ROOT=tmp_path,
        RAW_DATA_DIR="raw",
        PROCESSED_DATA_DIR="processed",
        INDEX_DATA_DIR="indexes",
        EVAL_DATA_DIR="eval",
    )


def _copy_fixture_collection(tmp_path: Path) -> Settings:
    """Copy the retrieval fixture manifests into a temporary collection path.

    Why this function exists:
        The retrieval service loads manifests from the same local filesystem
        layout used by the real indexing pipeline. Copying the fixture dataset
        into that layout lets the tests exercise the full retrieval loading
        flow rather than bypassing it with in-memory stubs.

    Parameters:
        tmp_path: Temporary directory that should receive the fixture
            collection.

    Returns:
        A ``Settings`` object pointing at the temporary collection root.

    Edge cases handled:
        Parent directories are created before copying so the tests remain
        stable even when the temporary directory starts empty.
    """
    settings = _build_settings(tmp_path)
    collection_directory = settings.resolve_index_data_path("kb")
    collection_directory.mkdir(parents=True, exist_ok=True)

    fixture_directory = Path(__file__).parent / "fixtures" / "retrieval"
    copyfile(fixture_directory / "vector_inputs.json", collection_directory / "vector_inputs.json")
    copyfile(fixture_directory / "lexical_inputs.json", collection_directory / "lexical_inputs.json")
    return settings


def test_retrieval_service_executes_hybrid_query_and_returns_provenance(tmp_path: Path) -> None:
    """Verify that hybrid retrieval merges vector and lexical scores into ranked results."""
    settings = _copy_fixture_collection(tmp_path)
    retrieval_service = RetrievalService.from_collection(
        collection_name="kb",
        embedding_service=HashEmbeddingService(),
        settings=settings,
        enable_hybrid=True,
    )

    # Execute a billing-focused query because lexical matching should strongly
    # favor the billing guide while the result still carries vector and merge
    # score details.
    results = retrieval_service.retrieve(
        RetrievalRequest(query="billing invoice statements", top_k=2, min_score=0.1)
    )

    # Assert that the billing guide ranks first and that the result carries the
    # public metadata and provenance required by the retrieval contract.
    assert results
    assert results[0].chunk_id == "doc-2:0"
    assert results[0].title == "Billing Portal Guide"
    assert results[0].document_metadata["source_uri"] == "kb://billing-portal"
    assert results[0].provenance["retrieval_mode"] == "hybrid"
    assert results[0].provenance["merge_strategy"] == "weighted_normalized_sum"
    assert results[0].metadata["lexical_score"] > 0.0
    assert results[0].citation.locator == "Section: Invoices"


def test_retrieval_service_supports_vector_only_queries(tmp_path: Path) -> None:
    """Verify that retrieval still works when the lexical backend is disabled."""
    settings = _copy_fixture_collection(tmp_path)
    retrieval_service = RetrievalService.from_collection(
        collection_name="kb",
        embedding_service=HashEmbeddingService(),
        settings=settings,
        enable_hybrid=False,
    )

    # Run an auth-related query using vector-only retrieval because the service
    # should still embed the query and score semantic candidates cleanly.
    results = retrieval_service.retrieve(RetrievalRequest(query="reset login password", top_k=2))

    # Assert that the password guide is returned and that provenance reports
    # the vector-only retrieval mode.
    assert results[0].chunk_id == "doc-1:0"
    assert results[0].provenance["retrieval_mode"] == "vector"
    assert results[0].metadata["vector_score"] >= results[0].metadata["lexical_score"]


def test_retrieval_service_applies_filters_and_archive_handling(tmp_path: Path) -> None:
    """Verify that retrieval filters candidates before ranking and merging."""
    settings = _copy_fixture_collection(tmp_path)
    retrieval_service = RetrievalService.from_collection(
        collection_name="kb",
        embedding_service=HashEmbeddingService(),
        settings=settings,
        enable_hybrid=True,
    )

    # Request support-ticket results while archived content remains excluded, so
    # the archived support ticket should be filtered out before scoring.
    results = retrieval_service.retrieve(
        RetrievalRequest(query="browser cache sign in", top_k=2, source_types=["support_ticket"])
    )

    # Assert that archive filtering happens before result construction and only
    # the explicitly allowed archived query returns the ticket.
    assert results == []

    archived_results = retrieval_service.retrieve(
        RetrievalRequest(
            query="browser cache sign in",
            top_k=2,
            source_types=["support_ticket"],
            include_archived=True,
        )
    )

    assert archived_results[0].chunk_id == "ticket-9:0"
    assert archived_results[0].source_type == "support_ticket"
