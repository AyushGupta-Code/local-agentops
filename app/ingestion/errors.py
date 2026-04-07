"""Ingestion-specific exceptions used to classify source loading failures."""


class IngestionError(Exception):
    """Base exception for ingestion failures."""


class UnsupportedFormatError(IngestionError):
    """Raised when a discovered file uses an unsupported extension."""


class SourceLoadError(IngestionError):
    """Raised when a supported file cannot be loaded safely."""
