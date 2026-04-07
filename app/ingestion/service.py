"""Source discovery and registration service for ingestion."""

from pathlib import Path

from app.ingestion.discovery import discover_files
from app.ingestion.errors import IngestionError, SourceLoadError, UnsupportedFormatError
from app.ingestion.loaders import load_raw_documents_from_file
from app.ingestion.metadata import is_supported_file
from app.ingestion.models import IngestionFileResult, IngestionIssue, IngestionReport
from app.models.schemas import RawDocument


class IngestionService:
    """Service for scanning directories and registering raw source records."""

    def ingest_directories(self, input_directories: list[Path], *, recursive: bool = True) -> IngestionReport:
        """Scan input directories and ingest supported source files.

        Why this function exists:
            The ingestion pipeline begins with filesystem scanning. This method
            coordinates discovery, file filtering, per-file loading, and error
            aggregation so later parsing stages receive a clean batch of
            ``RawDocument`` records.

        Parameters:
            input_directories: One or more filesystem directories that should be
                scanned for source files.
            recursive: Whether nested subdirectories should be included in the
                discovery pass.

        Returns:
            An ``IngestionReport`` containing discovered files, registered raw
            documents, and any issues encountered along the way.

        Edge cases handled:
            Missing directories are ignored by discovery, unsupported files are
            recorded as issues instead of stopping the run, and corrupt files
            produce file-level errors without aborting the rest of the batch.
        """
        # Normalize the caller-provided paths first so the resulting report is
        # stable and independent from the shell's working directory.
        scanned_paths = [path.resolve(strict=False) for path in input_directories]

        # Discover all concrete files beneath the scan roots before format
        # filtering so the report can describe the full scope of the run.
        discovered_files = discover_files(scanned_paths, recursive=recursive)

        # Process each discovered file independently so one bad file cannot
        # block the rest of the ingestion batch.
        file_results = [self.ingest_file(path) for path in discovered_files]

        # Return the aggregated ingestion report to the caller.
        return IngestionReport(
            scanned_paths=scanned_paths,
            discovered_files=discovered_files,
            file_results=file_results,
        )

    def ingest_file(self, path: Path) -> IngestionFileResult:
        """Ingest one discovered file into one or more raw-document records.

        Why this function exists:
            Each file should be processed in isolation so the batch service can
            continue when one file is corrupt, unsupported, or otherwise broken.

        Parameters:
            path: Filesystem path of the discovered source file.

        Returns:
            An ``IngestionFileResult`` containing any registered documents and
            any issues recorded for the file.

        Edge cases handled:
            Unsupported file types become clean issues; corrupt supported files
            produce issues rather than uncaught exceptions; successful files
            return populated document lists with no issues.
        """
        # Reject unsupported files early so the report records that the file was
        # seen but intentionally not ingested.
        if not is_supported_file(path):
            return IngestionFileResult(
                path=path,
                documents=[],
                issues=[
                    IngestionIssue(
                        path=path,
                        reason="unsupported_format",
                        detail=f"Unsupported file extension: {path.suffix or '[none]'}",
                    )
                ],
            )

        try:
            # Load the file into one or more raw-document records using the
            # shared format-dispatch loader.
            documents = load_raw_documents_from_file(path)
        except UnsupportedFormatError as error:
            # Convert unsupported-format exceptions into reportable issues so
            # the batch can continue processing other files.
            return self._build_issue_result(path, reason="unsupported_format", detail=str(error))
        except SourceLoadError as error:
            # Capture file-level load failures without aborting the overall run.
            return self._build_issue_result(path, reason="source_load_error", detail=str(error))
        except IngestionError as error:
            # Preserve a broad ingestion-specific failure path for future loader
            # helpers that may introduce more specialized exception types.
            return self._build_issue_result(path, reason="ingestion_error", detail=str(error))

        # Return the successful file result containing the registered documents.
        return IngestionFileResult(path=path, documents=documents, issues=[])

    def ingest_document(self, document: RawDocument) -> RawDocument:
        """Return an already-built raw document unchanged.

        Why this function exists:
            Some future connectors may bypass filesystem discovery and produce
            ``RawDocument`` instances directly. Keeping this method preserves a
            simple service boundary for those cases.

        Parameters:
            document: Raw source payload already converted into the shared
                internal schema.

        Returns:
            The same ``RawDocument`` instance received by the method.

        Edge cases handled:
            There are no special edge cases yet because the method is a
            pass-through placeholder for direct-ingest connectors.
        """
        # Return the provided document unchanged because persistence and
        # deduplication are not part of the current ingestion stage.
        return document

    def _build_issue_result(self, path: Path, *, reason: str, detail: str) -> IngestionFileResult:
        """Build a standardized file-result object for ingestion failures.

        Why this function exists:
            The service records several categories of file-level failures. This
            helper keeps the resulting report structure consistent across all
            error branches.

        Parameters:
            path: Filesystem path of the file that failed.
            reason: Short machine-readable reason code describing the failure.
            detail: Detailed human-readable explanation of the issue.

        Returns:
            An ``IngestionFileResult`` containing no documents and one issue.

        Edge cases handled:
            The helper always returns a valid result object, even when the error
            detail came from an unexpected loader failure.
        """
        # Construct a consistent failure payload so callers can inspect issues
        # uniformly regardless of the specific loader branch that failed.
        return IngestionFileResult(
            path=path,
            documents=[],
            issues=[IngestionIssue(path=path, reason=reason, detail=detail)],
        )
