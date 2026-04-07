"""Typed models for vector and lexical index preparation."""

from pathlib import Path

from pydantic import BaseModel, Field


class VectorIndexInput(BaseModel):
    """Represents one chunk prepared for vector embedding."""

    chunk_id: str = Field(description="Chunk identifier used to join embeddings back to source chunks.")
    document_id: str = Field(description="Parent document identifier for grouping and filtering.")
    text: str = Field(description="Chunk text that should be embedded by a vector backend.")
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Chunk provenance and retrieval metadata carried into the vector index.",
    )


class LexicalIndexInput(BaseModel):
    """Represents one chunk prepared for lexical search indexing."""

    chunk_id: str = Field(description="Chunk identifier used for lexical result joins.")
    document_id: str = Field(description="Parent document identifier for filtering and aggregation.")
    text: str = Field(description="Normalized text to include in a lexical index.")
    terms: list[str] = Field(
        default_factory=list,
        description="Simple pre-tokenized terms useful for lexical backends or debugging.",
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Provenance metadata preserved alongside lexical entries.",
    )


class IndexPreparationResult(BaseModel):
    """Aggregated result from preparing local vector and lexical index inputs."""

    chunk_count: int = Field(description="Number of chunk records included in the preparation batch.")
    vector_manifest_path: Path = Field(description="Path of the persisted vector-input manifest.")
    lexical_manifest_path: Path | None = Field(
        default=None,
        description="Path of the persisted lexical-input manifest when lexical preparation is enabled.",
    )
    metadata_manifest_path: Path = Field(description="Path of the indexing metadata manifest.")
