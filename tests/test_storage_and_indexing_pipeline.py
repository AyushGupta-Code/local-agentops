"""Unit tests for local persistence and index preparation."""

import json
from pathlib import Path

from app.indexing.service import IndexingService
from app.models.schemas import DocumentChunk, ParsedDocument
from app.storage.service import LocalFileStorageService


def _build_parsed_document() -> ParsedDocument:
    """Create a small parsed document fixture for persistence tests.

    Why this function exists:
        Multiple tests need a consistent parsed document object. Using one
        helper keeps the fixture compact and avoids repeating the same schema
        construction noise in every test.

    Parameters:
        This helper does not accept parameters because the tests only need one
        stable parsed-document fixture shape.

    Returns:
        A valid ``ParsedDocument`` instance suitable for storage tests.

    Edge cases handled:
        The fixture includes minimal but concrete metadata so load/save behavior
        exercises the normal storage shape rather than an unrealistic empty one.
    """
    return ParsedDocument(
        document_id="doc-1",
        source_type="document",
        title="Stored Document",
        plain_text="This is stored parsed text.",
        language="en",
        sections=[],
        source_uri="/tmp/doc-1.txt",
        metadata={"source_file": "/tmp/doc-1.txt"},
    )


def _build_chunks() -> list[DocumentChunk]:
    """Create a small chunk batch fixture for indexing tests.

    Why this function exists:
        Storage and indexing tests both need realistic chunk objects with
        provenance metadata. Keeping the fixture builder shared makes those
        tests easier to read and keep in sync.

    Parameters:
        This helper does not take parameters because the tests use one stable
        chunk batch shape.

    Returns:
        A list of valid ``DocumentChunk`` objects.

    Edge cases handled:
        The chunk metadata includes file and section provenance so index
        preparation can prove that metadata survives serialization.
    """
    return [
        DocumentChunk(
            chunk_id="doc-1:0:0:20",
            document_id="doc-1",
            source_type="document",
            text="This is the first chunk.",
            token_count_estimate=6,
            char_start=0,
            char_end=24,
            chunk_index=0,
            section_heading="Intro",
            metadata={"source_file": "/tmp/doc-1.txt", "section_title": "Intro"},
        ),
        DocumentChunk(
            chunk_id="doc-1:1:20:40",
            document_id="doc-1",
            source_type="document",
            text="This is the second chunk.",
            token_count_estimate=6,
            char_start=20,
            char_end=45,
            chunk_index=1,
            section_heading="Intro",
            metadata={"source_file": "/tmp/doc-1.txt", "section_title": "Intro"},
        ),
    ]


def test_local_storage_persists_and_loads_parsed_documents(tmp_path: Path) -> None:
    """Verify that parsed documents are stored locally and loaded back as schemas."""
    # Build settings rooted in the temporary directory so the test exercises the
    # real local storage behavior without touching repository state.
    from app.core.config import Settings

    settings = Settings(DATA_ROOT=tmp_path, RAW_DATA_DIR="raw", PROCESSED_DATA_DIR="processed", INDEX_DATA_DIR="indexes", EVAL_DATA_DIR="eval")
    storage = LocalFileStorageService(settings)
    document = _build_parsed_document()

    # Persist and reload the parsed document through the public storage API.
    write_result = storage.save_parsed_document(document)
    loaded_document = storage.load_parsed_document(document.document_id)

    # Assert that the document round-trips through JSON persistence correctly.
    assert write_result.record_count == 1
    assert write_result.path.exists()
    assert loaded_document.document_id == document.document_id
    assert loaded_document.metadata["source_file"] == "/tmp/doc-1.txt"


def test_local_storage_persists_and_loads_chunk_batches(tmp_path: Path) -> None:
    """Verify that chunk files are persisted locally and loaded back as schemas."""
    from app.core.config import Settings

    settings = Settings(DATA_ROOT=tmp_path, RAW_DATA_DIR="raw", PROCESSED_DATA_DIR="processed", INDEX_DATA_DIR="indexes", EVAL_DATA_DIR="eval")
    storage = LocalFileStorageService(settings)
    chunks = _build_chunks()

    # Persist and reload the chunk batch through the storage abstraction.
    write_result = storage.save_chunks("doc-1", chunks)
    loaded_chunks = storage.load_chunks("doc-1")

    # Assert that the chunk list round-trips and keeps provenance metadata.
    assert write_result.record_count == 2
    assert len(loaded_chunks) == 2
    assert loaded_chunks[0].metadata["source_file"] == "/tmp/doc-1.txt"


def test_indexing_service_prepares_vector_and_lexical_manifests(tmp_path: Path) -> None:
    """Verify that the indexing service writes local vector and lexical manifests."""
    from app.core.config import Settings

    settings = Settings(DATA_ROOT=tmp_path, RAW_DATA_DIR="raw", PROCESSED_DATA_DIR="processed", INDEX_DATA_DIR="indexes", EVAL_DATA_DIR="eval")
    storage = LocalFileStorageService(settings)
    indexing_service = IndexingService(storage=storage, settings=settings)
    chunks = _build_chunks()

    # Run full index preparation so vector, lexical, and metadata manifests are
    # all written under the managed index directory.
    result = indexing_service.index("doc-1", chunks, collection_name="kb", include_lexical=True)

    # Assert that the manifest files were created and contain the expected
    # number of prepared records.
    assert result.chunk_count == 2
    assert result.vector_manifest_path.exists()
    assert result.lexical_manifest_path is not None and result.lexical_manifest_path.exists()
    assert result.metadata_manifest_path.exists()

    vector_payload = json.loads(result.vector_manifest_path.read_text(encoding="utf-8"))
    lexical_payload = json.loads(result.lexical_manifest_path.read_text(encoding="utf-8"))
    metadata_payload = json.loads(result.metadata_manifest_path.read_text(encoding="utf-8"))

    assert len(vector_payload["records"]) == 2
    assert len(lexical_payload["records"]) == 2
    assert lexical_payload["records"][0]["terms"][0] == "this"
    assert vector_payload["records"][0]["metadata"]["source_type"] == "document"
    assert lexical_payload["records"][0]["metadata"]["chunk_index"] == 0
    assert metadata_payload["collection_name"] == "kb"


def test_indexing_service_can_skip_lexical_preparation(tmp_path: Path) -> None:
    """Verify that lexical manifest generation is optional."""
    from app.core.config import Settings

    settings = Settings(DATA_ROOT=tmp_path, RAW_DATA_DIR="raw", PROCESSED_DATA_DIR="processed", INDEX_DATA_DIR="indexes", EVAL_DATA_DIR="eval")
    storage = LocalFileStorageService(settings)
    indexing_service = IndexingService(storage=storage, settings=settings)

    # Prepare indexes without lexical output because the service should support
    # vector-only workflows cleanly.
    result = indexing_service.prepare_indexes(_build_chunks(), collection_name="vector-only", include_lexical=False)

    # Assert that vector and metadata manifests exist while lexical output is
    # intentionally absent.
    assert result.vector_manifest_path.exists()
    assert result.lexical_manifest_path is None
    assert result.metadata_manifest_path.exists()
