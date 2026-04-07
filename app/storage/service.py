"""Local-first file storage implementation for parsed documents and chunks."""

import json
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.models.schemas import DocumentChunk, ParsedDocument
from app.storage.models import StorageWriteResult


class LocalFileStorageService:
    """Filesystem-backed storage service for parsed documents, chunks, and metadata."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the local storage service with managed repository paths.

        Why this function exists:
            The project is explicitly local-first, so the default persistence
            shape should be simple JSON files inside managed repository
            directories. The constructor centralizes those directory decisions so
            later code does not hardcode paths everywhere.

        Parameters:
            settings: Optional validated application settings instance. When not
                provided, the shared process settings are used.

        Returns:
            This initializer does not return a value; it stores the managed
            storage directories on the service instance.

        Edge cases handled:
            Missing directories are created eagerly through the settings object,
            which reduces partial-write failures caused by absent parent paths.
        """
        self._settings = settings or get_settings()
        directories = self._settings.create_data_directories()
        self._processed_directory = directories["processed"]
        self._index_directory = directories["indexes"]

    def resolve_storage_path(self, relative_path: str) -> Path:
        """Resolve a safe path under the processed-data directory.

        Why this function exists:
            Storage callers occasionally need a managed path for future file
            writes or reads. Exposing one resolver keeps path traversal checks in
            the settings layer instead of spreading them across callers.

        Parameters:
            relative_path: Logical path fragment beneath the processed-data
                directory.

        Returns:
            An absolute path under the processed-data directory.

        Edge cases handled:
            Absolute and escaping paths are rejected by the shared settings
            resolver before any filesystem operation occurs.
        """
        return self._settings.resolve_processed_data_path(relative_path)

    def save_parsed_document(self, document: ParsedDocument) -> StorageWriteResult:
        """Persist one parsed document as a JSON file.

        Why this function exists:
            Parsed documents need to be recoverable without rerunning parsing,
            especially for local debugging and incremental indexing. Storing one
            document per file also keeps failures isolated and merges simple.

        Parameters:
            document: Parsed document to serialize to local storage.

        Returns:
            A ``StorageWriteResult`` describing the written file and record
            count.

        Edge cases handled:
            Parent directories are created automatically, and serialization
            errors surface immediately instead of being swallowed.
        """
        path = self.resolve_storage_path(f"parsed/{document.document_id}.json")
        self._write_json_file(path, document.model_dump(mode="json"))
        return StorageWriteResult(path=path, record_count=1, storage_kind="parsed_document")

    def save_chunks(self, document_id: str, chunks: list[DocumentChunk]) -> StorageWriteResult:
        """Persist chunk records for one document as a JSON array.

        Why this function exists:
            The indexing pipeline needs stable local copies of chunk records so
            vector and lexical preparation can be rerun without rechunking.
            Grouping chunks by document keeps writes localized and easy to
            inspect.

        Parameters:
            document_id: Identifier of the document whose chunks are being
                persisted.
            chunks: Retrieval-ready chunk records to serialize.

        Returns:
            A ``StorageWriteResult`` describing the written chunk file and count.

        Edge cases handled:
            Empty chunk lists are still written as an empty JSON array so the
            persisted state remains explicit rather than ambiguous.
        """
        path = self.resolve_storage_path(f"chunks/{document_id}.json")
        payload = [chunk.model_dump(mode="json") for chunk in chunks]
        self._write_json_file(path, payload)
        return StorageWriteResult(path=path, record_count=len(chunks), storage_kind="chunks")

    def save_metadata(self, relative_path: str, metadata: dict[str, object]) -> StorageWriteResult:
        """Persist arbitrary metadata under the index-data directory.

        Why this function exists:
            Index preparation produces manifests and backend-specific inputs that
            are not themselves parsed documents or chunks. This method provides
            one generic metadata persistence path without binding the codebase to
            a database schema prematurely.

        Parameters:
            relative_path: Logical file path beneath the managed index-data
                directory.
            metadata: JSON-serializable metadata payload to persist.

        Returns:
            A ``StorageWriteResult`` describing the metadata file write.

        Edge cases handled:
            Parent directories are created automatically, and callers get an
            explicit exception if the payload is not JSON serializable.
        """
        path = self._settings.resolve_index_data_path(relative_path)
        self._write_json_file(path, metadata)
        return StorageWriteResult(path=path, record_count=1, storage_kind="metadata")

    def load_parsed_document(self, document_id: str) -> ParsedDocument:
        """Load one parsed document from local storage.

        Why this function exists:
            Local indexing and debugging flows often need to resume from parsed
            output already persisted on disk. This loader reconstructs the shared
            schema object from that stored JSON.

        Parameters:
            document_id: Identifier of the persisted parsed document to load.

        Returns:
            The reconstructed ``ParsedDocument`` instance.

        Edge cases handled:
            Missing files raise ``FileNotFoundError`` naturally so callers can
            decide whether to recover or stop.
        """
        path = self.resolve_storage_path(f"parsed/{document_id}.json")
        payload = self._read_json_file(path)
        return ParsedDocument.model_validate(payload)

    def load_chunks(self, document_id: str) -> list[DocumentChunk]:
        """Load persisted chunk records for one document.

        Why this function exists:
            Vector and lexical index preparation may be rerun independently of
            the chunking step. This loader reconstructs chunk objects from the
            local chunk store for one document.

        Parameters:
            document_id: Identifier of the document whose chunks should be
                loaded.

        Returns:
            A list of reconstructed ``DocumentChunk`` instances.

        Edge cases handled:
            Missing files raise ``FileNotFoundError`` so callers do not silently
            treat absent chunk data as an empty document.
        """
        path = self.resolve_storage_path(f"chunks/{document_id}.json")
        payload = self._read_json_file(path)
        return [DocumentChunk.model_validate(item) for item in payload]

    def load_all_chunks(self) -> list[DocumentChunk]:
        """Load all chunk files stored under the processed-data directory.

        Why this function exists:
            Batch indexing runners usually need the entire local chunk corpus.
            This helper gathers every persisted chunk file into one flat list
            while preserving the file-backed storage abstraction.

        Parameters:
            This method reads the managed chunk directory configured on the
            service instance and therefore accepts no explicit parameters.

        Returns:
            A flat list of all stored ``DocumentChunk`` records.

        Edge cases handled:
            If the chunk directory exists but contains no files, the method
            returns an empty list instead of failing.
        """
        chunk_directory = self.resolve_storage_path("chunks")
        if not chunk_directory.exists():
            return []

        all_chunks: list[DocumentChunk] = []
        for path in sorted(chunk_directory.glob("*.json")):
            payload = self._read_json_file(path)
            all_chunks.extend(DocumentChunk.model_validate(item) for item in payload)
        return all_chunks

    def _write_json_file(self, path: Path, payload: Any) -> None:
        """Write JSON payloads to disk atomically enough for local workflows.

        Why this function exists:
            Parsed documents, chunks, and metadata all serialize to JSON. Using
            one write helper keeps formatting consistent and makes future atomic
            write upgrades straightforward.

        Parameters:
            path: Absolute destination file path under managed storage.
            payload: JSON-serializable Python object to persist.

        Returns:
            This helper does not return a value; it writes the payload to disk.

        Edge cases handled:
            Parent directories are created automatically, and write failures
            surface immediately so callers can react rather than assuming
            persistence succeeded.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=True)

    def _read_json_file(self, path: Path) -> Any:
        """Read JSON payloads from local storage.

        Why this function exists:
            Storage loads for parsed documents and chunks share the same JSON
            decoding behavior. Centralizing it keeps read semantics consistent
            and isolates future validation or migration logic.

        Parameters:
            path: Absolute file path of the JSON payload to read.

        Returns:
            The decoded Python object stored in the file.

        Edge cases handled:
            Missing files and JSON decode failures are not swallowed because the
            caller needs an explicit signal that persisted state is inconsistent
            or absent.
        """
        with path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
