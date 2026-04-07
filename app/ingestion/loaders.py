"""Format-specific loaders for the ingestion pipeline."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.ingestion.errors import SourceLoadError, UnsupportedFormatError
from app.ingestion.metadata import (
    build_document_id,
    build_source_metadata,
    get_file_extension,
    get_ingestion_timestamp,
    infer_source_type,
)
from app.models.schemas import RawDocument


def _extract_text_document(path: Path) -> str:
    """Read plain-text content from TXT or Markdown files.

    Why this function exists:
        Text and Markdown files can already be represented as a single raw
        document without deeper parsing. This helper keeps that straightforward
        file-reading logic out of the higher-level service.

    Parameters:
        path: Filesystem path of the text-like source file.

    Returns:
        The decoded text content of the source file.

    Edge cases handled:
        UTF-8 decode errors are retried with replacement characters so mildly
        malformed text files can still be registered instead of failing the
        entire ingestion run.
    """
    try:
        # Read the file as UTF-8 text first because that is the most common
        # encoding for local documentation and exported notes.
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Retry with replacement characters so a mostly readable file is still
        # registered even when it contains a few invalid byte sequences.
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        # Wrap filesystem read failures in a source-load error so the caller can
        # record a clean ingestion issue without crashing the whole batch.
        raise SourceLoadError(f"Unable to read text file: {error}") from error


def _extract_pdf_placeholder_content(path: Path) -> str:
    """Validate a PDF file and return placeholder content for later parsing.

    Why this function exists:
        PDF ingestion in this stage is about discovery and registration, not
        deep text extraction. The helper confirms the file looks like a PDF and
        stores a placeholder content string so later parsing can take over.

    Parameters:
        path: Filesystem path of the PDF source file.

    Returns:
        A placeholder content string indicating that the binary source was
        registered for later parsing.

    Edge cases handled:
        Files with an invalid PDF header or unreadable bytes raise a
        ``SourceLoadError`` so the service can record them as corrupt.
    """
    try:
        # Read the first few bytes only because this stage needs lightweight
        # validation rather than full binary parsing.
        header = path.read_bytes()[:5]
    except OSError as error:
        # Surface a consistent ingestion-specific error when the file cannot be
        # opened or read from disk.
        raise SourceLoadError(f"Unable to read PDF file: {error}") from error

    # Confirm the source begins with the standard PDF header bytes so clearly
    # corrupt or mislabeled files are rejected early.
    if header != b"%PDF-":
        raise SourceLoadError("Invalid PDF header; file may be corrupt or mislabeled")

    # Return placeholder content because binary PDF text extraction belongs in
    # the later parsing stage, not the ingestion stage.
    return f"PDF source registered for later parsing: {path.name}"


def _parse_datetime(value: Any) -> datetime | None:
    """Convert loose tabular timestamp values into ``datetime`` objects.

    Why this function exists:
        CSV and JSON datasets often store timestamps as strings in multiple
        formats. This helper performs a small amount of normalization so row
        loaders can populate ``RawDocument`` timestamps consistently.

    Parameters:
        value: Candidate timestamp value extracted from a CSV or JSON row.

    Returns:
        A parsed ``datetime`` when the value is recognizable, otherwise
        ``None``.

    Edge cases handled:
        Empty values, unsupported formats, and non-string inputs simply return
        ``None`` instead of raising, allowing ingestion to proceed.
    """
    # Return ``None`` for missing values because not every dataset includes
    # creation or update timestamps.
    if value is None:
        return None

    # Preserve pre-parsed datetime objects unchanged for callers that already
    # normalized their source data.
    if isinstance(value, datetime):
        return value

    # Ignore non-string values that cannot be interpreted safely as timestamps.
    if not isinstance(value, str):
        return None

    normalized_value = value.strip()
    if not normalized_value:
        return None

    try:
        # Support ISO 8601 values and common ``Z`` suffixes emitted by APIs.
        return datetime.fromisoformat(normalized_value.replace("Z", "+00:00"))
    except ValueError:
        # Return ``None`` instead of failing the whole row when a timestamp is
        # malformed or in an unsupported format.
        return None


def _stringify_content(value: Any) -> str:
    """Convert row content fields into a stable string representation.

    Why this function exists:
        Tabular datasets may store message bodies as strings, numbers, or nested
        JSON-like structures. This helper ensures the ``RawDocument.content``
        field always receives concrete text.

    Parameters:
        value: Row field chosen as the source content.

    Returns:
        A non-empty string representation when possible, otherwise an empty
        string.

    Edge cases handled:
        ``None`` becomes an empty string; dictionaries and lists are serialized
        as compact JSON for traceability.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value).strip()


def _build_row_document(path: Path, row_data: dict[str, Any], *, record_index: int, connector_name: str) -> RawDocument:
    """Convert one CSV or JSON row into a concrete ``RawDocument``.

    Why this function exists:
        Ticket datasets often arrive as row-based exports rather than one file
        per ticket. This helper isolates the field mapping and metadata
        construction needed to turn each row into the project's shared internal
        raw-document contract.

    Parameters:
        path: Filesystem path of the dataset file currently being processed.
        row_data: Dictionary representing one source row or object from the
            dataset.
        record_index: Zero-based row position used for stable IDs and metadata.
        connector_name: Connector label recorded on the resulting document.

    Returns:
        A populated ``RawDocument`` ready for later parsing stages.

    Edge cases handled:
        Missing title or content fields fall back to ticket-style columns or
        synthetic values; absent timestamps and optional ticket fields remain
        ``None`` without failing the whole row.
    """
    # Infer the source type from both the file name and row columns so tabular
    # ticket exports can be classified without deep parsing.
    source_type = infer_source_type(path, row_data)

    # Use common row field names in priority order so the loader can handle a
    # range of CSV and JSON export shapes without bespoke mappers.
    title = (
        _stringify_content(row_data.get("title"))
        or _stringify_content(row_data.get("subject"))
        or _stringify_content(row_data.get("name"))
        or f"{source_type.replace('_', ' ').title()} Record {record_index + 1}"
    )
    content = (
        _stringify_content(row_data.get("content"))
        or _stringify_content(row_data.get("body"))
        or _stringify_content(row_data.get("description"))
        or _stringify_content(row_data.get("message"))
        or _stringify_content(row_data)
    )

    # Build baseline file and row provenance metadata that later stages can use
    # for traceability and debugging.
    ingestion_timestamp = get_ingestion_timestamp()
    metadata = build_source_metadata(
        path,
        source_type=source_type,
        ingestion_timestamp=ingestion_timestamp,
        record_index=record_index,
    )
    metadata["row_data"] = row_data

    # Construct the concrete raw document using support-ticket fields when they
    # are available from the row.
    try:
        # Validate the mapped row against the shared raw-document schema so the
        # ingestion pipeline catches malformed dataset rows early.
        document = RawDocument(
            document_id=build_document_id(path, record_index),
            external_id=_stringify_content(row_data.get("ticket_id"))
            or _stringify_content(row_data.get("id"))
            or None,
            source_type=source_type,
            connector_name=connector_name,
            title=title,
            content=content,
            mime_type="text/csv-row" if connector_name == "csv_import" else "application/json-row",
            source_uri=str(path.resolve(strict=False)),
            language=_stringify_content(row_data.get("language")) or "en",
            created_at=_parse_datetime(row_data.get("created_at")),
            updated_at=_parse_datetime(row_data.get("updated_at")),
            author_name=_stringify_content(row_data.get("author_name"))
            or _stringify_content(row_data.get("requester"))
            or None,
            customer_id=_stringify_content(row_data.get("customer_id")) or None,
            ticket_status=_stringify_content(row_data.get("status")) or None,
            priority=_stringify_content(row_data.get("priority")) or None,
            tags=[
                _stringify_content(tag)
                for tag in (
                    row_data.get("tags")
                    if isinstance(row_data.get("tags"), list)
                    else str(row_data.get("tags", "")).split(",")
                )
            ],
            metadata=metadata,
        )
    except ValidationError as error:
        # Wrap schema validation failures so the batch service can record the
        # bad row cleanly instead of crashing the whole ingestion run.
        raise SourceLoadError(f"Invalid row data in {path.name} at index {record_index}: {error}") from error

    # Return the mapped raw document to the caller.
    return document


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Load tabular row dictionaries from a CSV file.

    Why this function exists:
        CSV exports are a common format for support-ticket datasets. This helper
        keeps row discovery and CSV decoding out of the higher-level ingestion
        service.

    Parameters:
        path: Filesystem path of the CSV source file.

    Returns:
        A list of row dictionaries produced by ``csv.DictReader``.

    Edge cases handled:
        Empty files raise a ``SourceLoadError`` because they do not contribute
        usable records; CSV decoding failures are wrapped in ingestion-specific
        errors.
    """
    try:
        # Open the file with universal newline handling so exported CSV files
        # from different systems decode consistently.
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            rows = list(reader)
    except UnicodeDecodeError as error:
        raise SourceLoadError(f"Unable to decode CSV file: {error}") from error
    except csv.Error as error:
        raise SourceLoadError(f"Unable to parse CSV file: {error}") from error
    except OSError as error:
        raise SourceLoadError(f"Unable to read CSV file: {error}") from error

    # Reject files that decode but contain no header or records because they do
    # not represent a meaningful dataset for ingestion.
    if not rows and path.stat().st_size > 0:
        raise SourceLoadError("CSV file did not contain readable rows")

    # Return the parsed row dictionaries to the caller.
    return rows


def _load_json_payload(path: Path) -> Any:
    """Load a JSON payload from disk.

    Why this function exists:
        JSON sources may represent either one document, one ticket, or a list
        of row-like records. Keeping JSON decoding isolated makes error handling
        and later format branching easier to test.

    Parameters:
        path: Filesystem path of the JSON source file.

    Returns:
        The decoded JSON payload, which may be a dictionary, list, string, or
        another JSON-supported type.

    Edge cases handled:
        Invalid JSON raises ``SourceLoadError`` instead of leaking raw decoder
        exceptions through the service layer.
    """
    try:
        # Read and decode the JSON source into native Python structures for
        # later row or document conversion.
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SourceLoadError(f"Invalid JSON file: {error}") from error
    except UnicodeDecodeError as error:
        raise SourceLoadError(f"Unable to decode JSON file: {error}") from error
    except OSError as error:
        raise SourceLoadError(f"Unable to read JSON file: {error}") from error


def load_raw_documents_from_file(path: Path) -> list[RawDocument]:
    """Load one supported source file into one or more ``RawDocument`` records.

    Why this function exists:
        The ingestion service needs a single format-dispatch function that
        converts each discovered file into the shared raw-document schema while
        keeping format-specific details hidden behind modular helpers.

    Parameters:
        path: Filesystem path of the supported source file to ingest.

    Returns:
        A list of ``RawDocument`` records produced from the source file. Most
        files yield one record, while CSV and some JSON datasets may yield many.

    Edge cases handled:
        Unsupported formats raise ``UnsupportedFormatError``; corrupt or
        unreadable sources raise ``SourceLoadError`` so the caller can record a
        non-fatal issue and continue the batch.
    """
    extension = get_file_extension(path)
    connector_name = "filesystem"

    # Dispatch text-like single-document formats directly into one raw-document
    # record because they already map cleanly to the shared schema.
    if extension in {".txt", ".md"}:
        source_type = infer_source_type(path)
        ingestion_timestamp = get_ingestion_timestamp()
        try:
            # Build one raw-document record for the text file because text-like
            # inputs already map directly onto the shared schema.
            document = RawDocument(
                document_id=build_document_id(path),
                external_id=None,
                source_type=source_type,
                connector_name=connector_name,
                title=path.stem.replace("_", " ").replace("-", " ").strip() or path.name,
                content=_extract_text_document(path),
                mime_type="text/markdown" if extension == ".md" else "text/plain",
                source_uri=str(path.resolve(strict=False)),
                created_at=None,
                updated_at=None,
                metadata=build_source_metadata(
                    path,
                    source_type=source_type,
                    ingestion_timestamp=ingestion_timestamp,
                ),
            )
        except ValidationError as error:
            # Re-raise schema-level failures as loader errors so the service can
            # record them without terminating the batch.
            raise SourceLoadError(f"Invalid text document metadata for {path.name}: {error}") from error
        return [document]

    # Validate and register PDFs as binary document sources while deferring deep
    # parsing to the later parsing module.
    if extension == ".pdf":
        source_type = infer_source_type(path)
        ingestion_timestamp = get_ingestion_timestamp()
        try:
            # Register the PDF source with placeholder content while preserving
            # enough metadata for later binary parsing.
            document = RawDocument(
                document_id=build_document_id(path),
                external_id=None,
                source_type=source_type,
                connector_name=connector_name,
                title=path.stem.replace("_", " ").replace("-", " ").strip() or path.name,
                content=_extract_pdf_placeholder_content(path),
                mime_type="application/pdf",
                source_uri=str(path.resolve(strict=False)),
                created_at=None,
                updated_at=None,
                metadata=build_source_metadata(
                    path,
                    source_type=source_type,
                    ingestion_timestamp=ingestion_timestamp,
                ),
            )
        except ValidationError as error:
            # Re-wrap schema validation problems so the service can treat them
            # the same way as other file-level loader failures.
            raise SourceLoadError(f"Invalid PDF document metadata for {path.name}: {error}") from error
        return [document]

    # Convert each CSV row into a distinct raw-document record because ticket
    # exports usually represent one logical item per row.
    if extension == ".csv":
        rows = _load_csv_rows(path)
        return [
            _build_row_document(path, row_data, record_index=index, connector_name="csv_import")
            for index, row_data in enumerate(rows)
        ]

    # Branch JSON handling based on whether the payload represents one object,
    # many objects, or a simple text-like document.
    if extension == ".json":
        payload = _load_json_payload(path)
        if isinstance(payload, list):
            documents = [
                _build_row_document(path, row_data, record_index=index, connector_name="json_import")
                for index, row_data in enumerate(payload)
                if isinstance(row_data, dict)
            ]
            if not documents:
                raise SourceLoadError("JSON list must contain at least one object record")
            return documents
        if isinstance(payload, dict):
            if any(key in payload for key in ("content", "body", "description", "message", "subject", "title")):
                return [_build_row_document(path, payload, record_index=0, connector_name="json_import")]

            source_type = infer_source_type(path, payload)
            ingestion_timestamp = get_ingestion_timestamp()
            try:
                return [
                    RawDocument(
                        document_id=build_document_id(path),
                        external_id=None,
                        source_type=source_type,
                        connector_name="json_import",
                        title=path.stem.replace("_", " ").replace("-", " ").strip() or path.name,
                        content=json.dumps(payload, ensure_ascii=True),
                        mime_type="application/json",
                        source_uri=str(path.resolve(strict=False)),
                        created_at=None,
                        updated_at=None,
                        metadata=build_source_metadata(
                            path,
                            source_type=source_type,
                            ingestion_timestamp=ingestion_timestamp,
                        ),
                    )
                ]
            except ValidationError as error:
                # Convert schema validation problems into file-level loader
                # errors so the service can continue with the rest of the batch.
                raise SourceLoadError(f"Invalid JSON document metadata for {path.name}: {error}") from error
        raise SourceLoadError("JSON file must contain an object or list of objects")

    # Reject anything else explicitly so unsupported discovered files are
    # recorded as clean issues rather than silently ignored.
    raise UnsupportedFormatError(f"Unsupported file extension: {extension or '[none]'}")
