"""Storage backend abstractions for parsed documents and chunks."""

from pathlib import Path
from typing import Protocol

from app.models.schemas import DocumentChunk, ParsedDocument
from app.storage.models import StorageWriteResult


class DocumentStorageBackend(Protocol):
    """Protocol describing the persistence surface required by the pipeline."""

    def save_parsed_document(self, document: ParsedDocument) -> StorageWriteResult:
        """Persist one parsed document."""

    def save_chunks(self, document_id: str, chunks: list[DocumentChunk]) -> StorageWriteResult:
        """Persist chunk records for one document."""

    def save_metadata(self, relative_path: str, metadata: dict[str, object]) -> StorageWriteResult:
        """Persist arbitrary metadata related to parsing or indexing."""

    def load_parsed_document(self, document_id: str) -> ParsedDocument:
        """Load one parsed document from the backend."""

    def load_chunks(self, document_id: str) -> list[DocumentChunk]:
        """Load persisted chunks for one document."""

    def load_all_chunks(self) -> list[DocumentChunk]:
        """Load all persisted chunk records from the backend."""

    def resolve_storage_path(self, relative_path: str) -> Path:
        """Resolve a backend-managed storage path safely."""
