"""Parsing service that dispatches raw sources to format-specific parsers."""

from pathlib import Path

from app.core.logging import get_logger
from app.models.schemas import ParsedDocument, RawDocument
from app.parsing.errors import ParsingError
from app.parsing.parsers import (
    parse_json_document,
    parse_markdown_document,
    parse_pdf_document,
    parse_text_document,
    parse_ticket_document,
)
from app.parsing.pdf import PdfTextExtractor, PypdfTextExtractor


class ParsingService:
    """Service boundary for converting raw sources into normalized parsed documents."""

    def __init__(self, pdf_extractor: PdfTextExtractor | None = None) -> None:
        """Initialize the parsing service with a pluggable PDF extractor.

        Why this function exists:
            PDF extraction will likely evolve to include OCR or alternative
            libraries. Accepting the extractor as a dependency now keeps the
            service open for extension without changing its public API later.

        Parameters:
            pdf_extractor: Optional PDF page extractor implementation. When not
                supplied, the service uses the default ``pypdf`` backend.

        Returns:
            This initializer does not return a value; it configures parser
            dependencies on the service instance.

        Edge cases handled:
            Callers may pass a custom extractor in tests or future OCR flows,
            and the default backend is used otherwise.
        """
        # Store a logger on the service so parse failures are always emitted
        # explicitly before the exception is raised back to the caller.
        self._logger = get_logger(__name__)

        # Keep the PDF extractor swappable so OCR or another backend can be
        # added without rewriting the service dispatch logic.
        self._pdf_extractor = pdf_extractor or PypdfTextExtractor()

    def parse(self, document: RawDocument) -> ParsedDocument:
        """Parse one raw document into a normalized ``ParsedDocument``.

        Why this function exists:
            The rest of the application should not need to know which parser to
            call for each MIME type. This method centralizes dispatch and
            ensures parse failures are logged in one place.

        Parameters:
            document: Raw source artifact produced by ingestion.

        Returns:
            A normalized ``ParsedDocument`` ready for chunking and indexing.

        Edge cases handled:
            Unsupported MIME types raise ``ParsingError``; parser-specific
            failures are logged explicitly and re-raised so they are never
            swallowed silently.
        """
        try:
            # Route the document to the appropriate parser based on MIME type
            # and source type while preserving a single public service method.
            if document.mime_type == "application/pdf":
                return parse_pdf_document(document, extractor=self._pdf_extractor)
            if document.mime_type == "text/plain":
                return parse_text_document(document)
            if document.mime_type == "text/markdown":
                return parse_markdown_document(document)
            if document.source_type == "support_ticket" and document.mime_type in {
                "text/csv-row",
                "application/json-row",
            }:
                return parse_ticket_document(document)
            if document.mime_type == "application/json":
                return parse_json_document(document)
        except ParsingError:
            # Re-log parsing failures with document context before surfacing the
            # exception so operators can diagnose the failing source.
            self._logger.exception(
                "Failed to parse raw document",
                extra={
                    "document_id": document.document_id,
                    "source_uri": document.source_uri,
                    "mime_type": document.mime_type,
                },
            )
            raise

        # Reject unsupported MIME types explicitly so parsing behavior stays
        # predictable and easy to debug.
        error = ParsingError(
            f"Unsupported parsing strategy for mime_type={document.mime_type} "
            f"source_type={document.source_type}"
        )
        self._logger.error(
            "Unsupported parser input",
            extra={
                "document_id": document.document_id,
                "source_uri": document.source_uri,
                "mime_type": document.mime_type,
                "source_type": document.source_type,
            },
        )
        raise error

    def parse_many(self, documents: list[RawDocument]) -> list[ParsedDocument]:
        """Parse a batch of raw documents in order.

        Why this function exists:
            Batch-oriented pipeline stages often operate on lists of documents.
            This helper keeps ordering simple while reusing the same per-record
            dispatch and failure semantics as ``parse``.

        Parameters:
            documents: Ordered list of raw source records to parse.

        Returns:
            A list of parsed documents in the same order as the inputs.

        Edge cases handled:
            The method stops on the first parsing failure so broken sources are
            surfaced immediately rather than hidden inside partial batch output.
        """
        # Parse each document sequentially so provenance order is preserved and
        # any failure clearly points to the first problematic record.
        parsed_documents = [self.parse(document) for document in documents]

        # Return the parsed batch to the caller.
        return parsed_documents

    def parse_source_path(self, path: Path, *, title: str | None = None) -> ParsedDocument:
        """Parse a source file path directly for tests or utility scripts.

        Why this function exists:
            Some fixtures and maintenance scripts may want to exercise parsing
            without running the full ingestion pipeline first. This helper
            provides that convenience while still using the same parser logic.

        Parameters:
            path: Filesystem path of the source file to parse.
            title: Optional display title override used when constructing the
                temporary raw document wrapper.

        Returns:
            A normalized ``ParsedDocument`` for the supplied source file.

        Edge cases handled:
            The helper infers MIME type only for the formats already supported
            by the parsing layer and raises ``ParsingError`` for anything else.
        """
        # Infer a minimal raw-document wrapper around the file so the public
        # parsing service can be exercised directly by tests or scripts.
        suffix = path.suffix.lower()
        mime_type_map = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".pdf": "application/pdf",
            ".json": "application/json",
        }
        mime_type = mime_type_map.get(suffix)
        if mime_type is None:
            raise ParsingError(f"Unsupported direct parse format for {path.name}")

        raw_document = RawDocument(
            document_id=path.stem,
            source_type="document",
            connector_name="filesystem",
            title=title or path.stem,
            content=path.read_text(encoding="utf-8") if suffix != ".pdf" else "",
            mime_type=mime_type,
            source_uri=str(path.resolve(strict=False)),
            metadata={"source_path": str(path.resolve(strict=False))},
        )

        # Reuse the normal parse dispatch so the helper stays behaviorally
        # aligned with standard pipeline parsing.
        return self.parse(raw_document)
