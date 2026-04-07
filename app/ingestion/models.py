"""Typed ingestion result models used by the file-discovery pipeline."""

from pathlib import Path

from pydantic import BaseModel, Field

from app.models.schemas import RawDocument


class IngestionIssue(BaseModel):
    """Represents a single file-level ingestion failure or warning."""

    path: Path = Field(description="Filesystem path that produced the issue.")
    reason: str = Field(description="Short machine-readable or human-readable issue summary.")
    detail: str = Field(description="Detailed message describing what went wrong.")


class IngestionFileResult(BaseModel):
    """Represents the outcome of ingesting one discovered file."""

    path: Path = Field(description="Filesystem path that was processed.")
    documents: list[RawDocument] = Field(
        default_factory=list,
        description="Raw documents registered from this file, including one-per-row tabular records.",
    )
    issues: list[IngestionIssue] = Field(
        default_factory=list,
        description="Warnings or errors encountered while processing the file.",
    )


class IngestionReport(BaseModel):
    """Aggregated result returned after scanning one or more input directories."""

    scanned_paths: list[Path] = Field(
        default_factory=list,
        description="All filesystem paths that were considered during discovery.",
    )
    discovered_files: list[Path] = Field(
        default_factory=list,
        description="Concrete files discovered before format filtering and loading.",
    )
    file_results: list[IngestionFileResult] = Field(
        default_factory=list,
        description="Per-file results describing registered documents and issues.",
    )

    @property
    def documents(self) -> list[RawDocument]:
        """Flatten all registered raw documents into a single list.

        Why this function exists:
            Most downstream callers care about the combined batch of raw
            documents rather than the individual file boundaries. This property
            provides that convenience while still preserving the richer file
            report structure for debugging.

        Parameters:
            This property reads the already-populated ``file_results`` stored on
            the report instance, so it does not accept parameters.

        Returns:
            A flat list containing every ``RawDocument`` produced during the
            ingestion run.

        Edge cases handled:
            Files that failed ingestion contribute no documents, and an empty
            report simply returns an empty list.
        """
        # Walk each per-file result and gather every document into a single list
        # so downstream stages can consume a batch without manual flattening.
        documents = [document for file_result in self.file_results for document in file_result.documents]

        # Return the flattened document list to the caller.
        return documents

    @property
    def issues(self) -> list[IngestionIssue]:
        """Flatten all ingestion issues into a single list.

        Why this function exists:
            Batch ingestion should continue past individual file failures, but
            callers still need an easy way to inspect every warning and error at
            the end of the run.

        Parameters:
            This property operates on ``file_results`` already stored on the
            report instance, so there are no explicit parameters.

        Returns:
            A flat list containing every recorded ``IngestionIssue``.

        Edge cases handled:
            Successful files simply contribute no issues, and a fully clean
            report returns an empty list.
        """
        # Gather every file-level issue into a single list so the caller can
        # inspect failures without iterating nested report structures.
        issues = [issue for file_result in self.file_results for issue in file_result.issues]

        # Return the flattened issue list to the caller.
        return issues
