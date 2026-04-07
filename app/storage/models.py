"""Typed storage result models for local persistence operations."""

from pathlib import Path

from pydantic import BaseModel, Field


class StorageWriteResult(BaseModel):
    """Represents one successful storage write operation."""

    path: Path = Field(description="Filesystem path written by the storage backend.")
    record_count: int = Field(description="Number of logical records persisted to the path.")
    storage_kind: str = Field(description="Logical storage category such as parsed_document or chunks.")
