"""Metadata extraction helpers for ingestion source registration."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid5, NAMESPACE_URL


SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".csv", ".json"}


def get_ingestion_timestamp() -> datetime:
    """Return the current UTC timestamp for ingestion bookkeeping.

    Why this function exists:
        Each discovered source should carry a consistent timestamp showing when
        it entered the local pipeline. Using a dedicated helper makes the
        timestamp source explicit and easy to mock in tests.

    Parameters:
        This helper does not require parameters because it simply snapshots the
        current UTC time.

    Returns:
        A timezone-aware ``datetime`` representing the ingestion time.

    Edge cases handled:
        The function always returns UTC so callers do not need to reason about
        local timezone drift or naive datetimes.
    """
    # Capture the current UTC time so each registered record can be traced back
    # to the ingestion run that discovered it.
    timestamp = datetime.now(UTC)

    # Return the timestamp to the caller.
    return timestamp


def get_file_extension(path: Path) -> str:
    """Return a normalized lowercase file extension for a source path.

    Why this function exists:
        File loading decisions are based primarily on extension. Centralizing
        that normalization avoids subtle bugs from uppercase or mixed-case file
        suffixes encountered during directory scanning.

    Parameters:
        path: Filesystem path for the discovered source file.

    Returns:
        The lowercase extension string, including the leading period.

    Edge cases handled:
        Files without an extension return an empty string, which callers can
        treat as unsupported.
    """
    # Normalize the suffix to lowercase so format matching remains stable
    # regardless of how the source file was named.
    extension = path.suffix.lower()

    # Return the normalized extension to the caller.
    return extension


def is_supported_file(path: Path) -> bool:
    """Check whether a discovered file uses a supported ingestion format.

    Why this function exists:
        Directory scans may encounter unrelated files. This helper lets the
        discovery layer classify each file cleanly before the loader attempts to
        open it.

    Parameters:
        path: Filesystem path being evaluated during discovery.

    Returns:
        ``True`` when the file extension is one of the supported ingestion
        formats, otherwise ``False``.

    Edge cases handled:
        Unknown extensions and extensionless files both evaluate to ``False``.
    """
    # Look up the normalized extension in the supported-format set.
    supported = get_file_extension(path) in SUPPORTED_EXTENSIONS

    # Return the support decision to the caller.
    return supported


def infer_source_type(path: Path, row_data: dict[str, Any] | None = None) -> Literal["document", "support_ticket"]:
    """Infer whether a source should be treated as a document or support ticket.

    Why this function exists:
        The ingestion pipeline needs a concrete source type before later parsing
        and retrieval stages run. This helper uses lightweight heuristics based
        on the file name and optional tabular row fields so the design stays
        extensible without requiring deep parsing.

    Parameters:
        path: Filesystem path of the discovered source file.
        row_data: Optional row payload from CSV or JSON datasets that may expose
            ticket-specific fields such as ``ticket_id`` or ``status``.

    Returns:
        Either ``"document"`` or ``"support_ticket"``.

    Edge cases handled:
        Ambiguous files default to ``"document"``; ticket-like row keys or file
        names override that default when obvious support-ticket signals exist.
    """
    # Prepare a lowercase name string so filename heuristics are simple and
    # case-insensitive.
    lower_name = path.name.lower()

    # Treat files that explicitly look like ticket exports as support-ticket
    # datasets even before inspecting their row content.
    if "ticket" in lower_name or "zendesk" in lower_name or "support" in lower_name:
        return "support_ticket"

    # Inspect optional row data for common ticket-oriented columns that signal
    # support workflow content rather than general documents.
    if row_data:
        lower_keys = {str(key).strip().lower() for key in row_data}
        ticket_keys = {"ticket_id", "status", "priority", "requester", "customer_id", "subject"}
        if lower_keys.intersection(ticket_keys):
            return "support_ticket"

    # Default to document when no ticket-specific signals are present.
    return "document"


def get_file_created_timestamp(path: Path) -> datetime | None:
    """Best-effort retrieval of a file creation timestamp.

    Why this function exists:
        Source registration should capture the original file timing when the
        operating system provides it. This metadata helps later debugging and
        evaluation without requiring deep content parsing.

    Parameters:
        path: Filesystem path of the source file whose metadata should be read.

    Returns:
        A timezone-aware ``datetime`` if creation metadata is available,
        otherwise ``None``.

    Edge cases handled:
        Platforms that do not expose creation time fall back to ``None`` rather
        than inventing a value; unreadable stat metadata also results in
        ``None``.
    """
    try:
        # Read file metadata from the filesystem so the ingest record can
        # include best-effort origin timing when the platform supports it.
        stat_result = path.stat()
    except OSError:
        # Return ``None`` when filesystem metadata cannot be read because file
        # discovery should remain resilient to transient stat failures.
        return None

    # Prefer the platform-specific creation time attribute when available.
    created_timestamp = getattr(stat_result, "st_birthtime", None)
    if created_timestamp is None:
        return None

    # Convert the raw epoch timestamp into a timezone-aware UTC datetime.
    created_at = datetime.fromtimestamp(created_timestamp, tz=UTC)

    # Return the timestamp to the caller.
    return created_at


def build_document_id(path: Path, record_index: int | None = None) -> str:
    """Generate a stable internal document identifier from file provenance.

    Why this function exists:
        The ingestion layer needs deterministic IDs so later chunking,
        retrieval, and evaluation stages can trace records back to their source
        file and, for tabular data, their original row.

    Parameters:
        path: Filesystem path of the source file.
        record_index: Optional zero-based row index for files that expand into
            multiple records, such as CSV or JSON ticket datasets.

    Returns:
        A deterministic string identifier suitable for ``RawDocument``.

    Edge cases handled:
        Files that produce one document omit the row suffix; tabular files use a
        row suffix to keep record IDs distinct.
    """
    # Build a deterministic namespace string from the absolute source path and
    # optional record index so repeated ingestions of unchanged data stay stable.
    raw_identifier = str(path.resolve(strict=False))
    if record_index is not None:
        raw_identifier = f"{raw_identifier}#row:{record_index}"

    # Convert the provenance string into a deterministic UUID-based identifier.
    document_id = str(uuid5(NAMESPACE_URL, raw_identifier))

    # Return the generated ID to the caller.
    return document_id


def build_source_metadata(
    path: Path,
    *,
    source_type: Literal["document", "support_ticket"],
    ingestion_timestamp: datetime,
    record_index: int | None = None,
) -> dict[str, Any]:
    """Build standard source metadata captured during ingestion.

    Why this function exists:
        The ingestion pipeline should register the same baseline provenance
        fields for every source regardless of file format. This helper keeps the
        metadata contract consistent and easy to extend.

    Parameters:
        path: Filesystem path of the discovered source file.
        source_type: Concrete source category inferred for the file or row.
        ingestion_timestamp: Timestamp representing when the current run
            discovered the source.
        record_index: Optional zero-based row number for tabular datasets.

    Returns:
        A metadata dictionary containing file-level provenance fields used by
        downstream pipeline stages.

    Edge cases handled:
        Missing file creation time is stored as ``None``; row-driven datasets
        include a row index only when the caller provides one.
    """
    # Gather the baseline provenance fields that later stages can use for audit
    # trails, debugging, and UI display.
    metadata: dict[str, Any] = {
        "file_name": path.name,
        "extension": get_file_extension(path),
        "source_path": str(path.resolve(strict=False)),
        "file_created_at": get_file_created_timestamp(path),
        "ingested_at": ingestion_timestamp,
        "source_type": source_type,
    }

    # Attach row metadata only when the source file expanded into multiple
    # logical records, such as ticket exports.
    if record_index is not None:
        metadata["record_index"] = record_index

    # Return the standardized metadata payload to the caller.
    return metadata
