"""Chunking service that converts parsed documents into retrieval-ready chunks."""

from app.chunking.strategies import chunk_document_by_strategy
from app.core.config import get_settings
from app.models.schemas import DocumentChunk, ParsedDocument


class ChunkingService:
    """Service boundary for splitting parsed documents into deterministic chunks."""

    def __init__(self, *, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        """Initialize the chunking service with configurable window settings.

        Why this function exists:
            Different deployments may want different chunk sizes and overlap
            defaults. The service constructor exposes those controls while still
            falling back to validated application settings.

        Parameters:
            chunk_size: Optional maximum chunk size in characters. When omitted,
                the service uses the configured application default.
            chunk_overlap: Optional overlap size in characters. When omitted,
                the service uses the configured application default.

        Returns:
            This initializer does not return a value; it stores validated
            chunking configuration on the service instance.

        Edge cases handled:
            The constructor rejects overlap values greater than or equal to the
            chosen chunk size because that would create duplicate or non-
            advancing windows.
        """
        settings = get_settings()
        resolved_chunk_size = chunk_size or settings.default_chunk_size
        resolved_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.default_chunk_overlap

        if resolved_chunk_overlap >= resolved_chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._chunk_size = resolved_chunk_size
        self._chunk_overlap = resolved_chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Split one parsed document into retrieval-ready chunks.

        Why this function exists:
            The indexing stage needs stable chunk objects regardless of document
            type. This method delegates strategy selection and returns chunk
            objects that are immediately usable for indexing.

        Parameters:
            document: Parsed document produced by the parsing layer.

        Returns:
            A list of deterministic ``DocumentChunk`` objects in source order.

        Edge cases handled:
            Strategy dispatch handles plain text, sectioned documents, and
            support tickets differently while sharing the same output contract.
        """
        return chunk_document_by_strategy(
            document,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def chunk_many(self, documents: list[ParsedDocument]) -> list[DocumentChunk]:
        """Chunk a batch of parsed documents while preserving document order.

        Why this function exists:
            Batch ingestion and parsing flows often hand the chunking layer a
            list of parsed documents. This helper flattens the resulting chunks
            into a single list ready for indexing.

        Parameters:
            documents: Parsed documents that should be chunked in order.

        Returns:
            A flat list of ``DocumentChunk`` objects covering all input
            documents.

        Edge cases handled:
            Empty input returns an empty list; each document keeps its own
            deterministic chunk indexing and provenance fields.
        """
        chunks = [chunk for document in documents for chunk in self.chunk(document)]
        return chunks
