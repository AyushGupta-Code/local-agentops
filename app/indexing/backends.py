"""Backend abstractions for vector indexing."""

from typing import Protocol

from app.indexing.models import VectorIndexInput


class VectorIndexBackend(Protocol):
    """Protocol describing the vector-index write surface used by the pipeline."""

    def prepare(self, inputs: list[VectorIndexInput]) -> list[dict[str, object]]:
        """Transform vector inputs into backend-specific serializable records."""
