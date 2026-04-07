"""Hybrid retrieval service built on local vector and lexical manifests."""
from typing import Any

from app.core.config import Settings, get_settings
from app.models.schemas import Citation, RetrievalRequest, RetrievalResult
from app.retrieval.backends import ManifestLexicalIndex, ManifestVectorIndex, RetrievalCandidate
from app.retrieval.embeddings import EmbeddingService, HashEmbeddingService


class RetrievalService:
    """Service boundary for vector and hybrid chunk retrieval."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | None = None,
        vector_index: ManifestVectorIndex,
        lexical_index: ManifestLexicalIndex | None = None,
        settings: Settings | None = None,
        collection_name: str = "default",
        vector_weight: float = 0.7,
        lexical_weight: float = 0.3,
    ) -> None:
        """Store the pluggable dependencies used during query execution.

        Why this function exists:
            Retrieval orchestration should stay modular. The service itself only
            coordinates query embedding, candidate scoring, score merging, and
            result construction, while the embedding and index details remain
            injectable dependencies.

        Parameters:
            embedding_service: Embedding implementation used to embed incoming
                queries. When omitted, a deterministic local embedder is used.
            vector_index: Ready-to-query vector index for semantic matching.
            lexical_index: Optional lexical index that contributes BM25
                candidates during hybrid retrieval.
            settings: Optional validated application settings used for citation
                truncation defaults and path resolution.
            collection_name: Logical retrieval collection name included in
                provenance output.
            vector_weight: Weight assigned to the normalized vector score during
                candidate merging.
            lexical_weight: Weight assigned to the normalized lexical score
                during candidate merging.

        Returns:
            This initializer stores the retrieval dependencies on the service.

        Edge cases handled:
            Weight validation fails fast for negative inputs or a zero total so
            score merging cannot silently produce meaningless results.
        """
        if vector_weight < 0.0 or lexical_weight < 0.0:
            raise ValueError("retrieval weights must be non-negative")

        if vector_weight == 0.0 and lexical_weight == 0.0:
            raise ValueError("at least one retrieval weight must be positive")

        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or HashEmbeddingService()
        self._vector_index = vector_index
        self._lexical_index = lexical_index
        self._collection_name = collection_name
        self._vector_weight = vector_weight
        self._lexical_weight = lexical_weight

    @classmethod
    def from_collection(
        cls,
        *,
        collection_name: str,
        embedding_service: EmbeddingService | None = None,
        settings: Settings | None = None,
        enable_hybrid: bool = True,
        vector_weight: float = 0.7,
        lexical_weight: float = 0.3,
    ) -> "RetrievalService":
        """Build the retrieval service from persisted collection manifests.

        Why this function exists:
            CLI runners and tests should be able to stand up retrieval with one
            call from the artifacts produced by the indexing pipeline. This
            constructor keeps that assembly path explicit and repeatable.

        Parameters:
            collection_name: Logical collection whose manifests should be
                loaded.
            embedding_service: Optional query and chunk embedding
                implementation.
            settings: Optional validated settings object used to resolve the
                managed index paths.
            enable_hybrid: Whether lexical retrieval should be loaded when a
                lexical manifest exists.
            vector_weight: Merge weight for the normalized vector score.
            lexical_weight: Merge weight for the normalized lexical score.

        Returns:
            A configured ``RetrievalService`` bound to the requested
            collection.

        Edge cases handled:
            Missing lexical manifests only disable the lexical signal; missing
            vector manifests still raise because vector retrieval is the base
            engine and the service cannot operate without it.
        """
        resolved_settings = settings or get_settings()
        resolved_embedding_service = embedding_service or HashEmbeddingService()
        vector_manifest_path = resolved_settings.resolve_index_data_path(
            f"{collection_name}/vector_inputs.json"
        )
        lexical_manifest_path = resolved_settings.resolve_index_data_path(
            f"{collection_name}/lexical_inputs.json"
        )

        vector_index = ManifestVectorIndex.from_manifest_path(
            vector_manifest_path,
            resolved_embedding_service,
        )
        lexical_index = None
        if enable_hybrid and lexical_manifest_path.exists():
            lexical_index = ManifestLexicalIndex.from_manifest_path(lexical_manifest_path)

        return cls(
            embedding_service=resolved_embedding_service,
            vector_index=vector_index,
            lexical_index=lexical_index,
            settings=resolved_settings,
            collection_name=collection_name,
            vector_weight=vector_weight,
            lexical_weight=lexical_weight,
        )

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Execute the full retrieval flow for one query.

        Why this function exists:
            This is the service entry point. It embeds the query, retrieves
            vector candidates, optionally retrieves lexical candidates, merges
            those candidates with weighted normalized scores, filters on the
            final fused score, and converts the winning chunks into structured
            retrieval results with provenance and citation data.

        Parameters:
            request: Retrieval request containing the query text, score
                threshold, filters, and result limit.

        Returns:
            A ranked list of ``RetrievalResult`` objects.

        Edge cases handled:
            Empty indexes, low-scoring queries, disabled lexical retrieval, and
            score thresholds that exclude all candidates all return an empty
            list rather than raising inside the ranking flow.
        """
        # Query embedding is the first vector retrieval step. If the embedding
        # model produces a zero vector, vector scores naturally collapse toward
        # zero and the service still returns a deterministic empty or low-score
        # result set.
        query_embedding = self._embedding_service.embed_query(request.query)

        vector_candidates = self._vector_index.search(
            query_embedding,
            top_k=request.top_k,
            source_types=request.source_types,
            tag_filters=[tag.lower() for tag in request.tag_filters],
            include_archived=request.include_archived,
        )

        lexical_candidates: dict[str, RetrievalCandidate] = {}
        if self._lexical_index is not None:
            lexical_candidates = self._lexical_index.search(
                request.query,
                top_k=request.top_k,
                source_types=request.source_types,
                tag_filters=[tag.lower() for tag in request.tag_filters],
                include_archived=request.include_archived,
            )

        merged_candidates = self._merge_candidates(
            vector_candidates=vector_candidates,
            lexical_candidates=lexical_candidates,
        )

        results: list[RetrievalResult] = []
        for candidate_payload in merged_candidates:
            fused_score = float(candidate_payload["score"])
            if request.min_score is not None and fused_score < request.min_score:
                continue

            rank = len(results) + 1
            results.append(self._build_result(candidate_payload, rank=rank))
            if len(results) >= request.top_k:
                break

        return results

    def _merge_candidates(
        self,
        *,
        vector_candidates: dict[str, RetrievalCandidate],
        lexical_candidates: dict[str, RetrievalCandidate],
    ) -> list[dict[str, Any]]:
        """Merge vector and lexical candidates with a weighted normalized sum.

        Why this function exists:
            Vector cosine scores and lexical BM25 scores are not directly
            comparable. Each backend normalizes its scores into the same
            ``0..1`` range, and this method then applies a simple weighted sum
            across the union of candidates so hybrid retrieval remains easy to
            reason about and debug.

        Parameters:
            vector_candidates: Top vector candidates keyed by ``chunk_id``.
            lexical_candidates: Top lexical candidates keyed by ``chunk_id``.

        Returns:
            A descending list of merged candidate payloads containing the fused
            score, component scores, ranks, and the source record used to build
            the final retrieval result.

        Edge cases handled:
            Candidates that only appear in one backend receive a zero score from
            the missing backend, which allows the merge to stay deterministic
            without inventing absent evidence.
        """
        total_weight = self._vector_weight + self._lexical_weight
        merged_payloads: list[dict[str, Any]] = []

        for chunk_id in sorted(set(vector_candidates) | set(lexical_candidates)):
            vector_candidate = vector_candidates.get(chunk_id)
            lexical_candidate = lexical_candidates.get(chunk_id)
            record = (
                vector_candidate.record
                if vector_candidate is not None
                else lexical_candidate.record
                if lexical_candidate is not None
                else None
            )
            if record is None:
                continue

            vector_score = (
                vector_candidate.normalized_score if vector_candidate is not None else 0.0
            )
            lexical_score = (
                lexical_candidate.normalized_score if lexical_candidate is not None else 0.0
            )

            # Score handling is intentionally simple: both component scores are
            # already normalized, so the final score is just their weighted
            # average. This keeps hybrid ranking explainable for tests and
            # makes it obvious how changing a weight shifts the result order.
            fused_score = (
                (self._vector_weight * vector_score) + (self._lexical_weight * lexical_score)
            ) / total_weight

            merged_payloads.append(
                {
                    "record": record,
                    "score": fused_score,
                    "vector_candidate": vector_candidate,
                    "lexical_candidate": lexical_candidate,
                }
            )

        merged_payloads.sort(
            key=lambda payload: (
                float(payload["score"]),
                payload["vector_candidate"].normalized_score
                if payload["vector_candidate"] is not None
                else 0.0,
                payload["lexical_candidate"].normalized_score
                if payload["lexical_candidate"] is not None
                else 0.0,
            ),
            reverse=True,
        )
        return merged_payloads

    def _build_result(self, candidate_payload: dict[str, Any], *, rank: int) -> RetrievalResult:
        """Convert one merged candidate payload into the public result schema.

        Why this function exists:
            Ranking and scoring are internal implementation details, but callers
            need a stable retrieval result shape with chunk text, final score,
            document metadata, and provenance. This method translates the merged
            candidate into that public contract.

        Parameters:
            candidate_payload: Merged candidate data emitted by
                ``_merge_candidates``.
            rank: One-based result rank after final sorting.

        Returns:
            A fully populated ``RetrievalResult``.

        Edge cases handled:
            Missing optional metadata fields fall back to predictable defaults
            so result construction does not fail just because a manifest record
            is sparse.
        """
        record = candidate_payload["record"]
        metadata = dict(record.metadata)
        source_type = str(metadata.get("source_type", "document"))
        title = str(metadata.get("title", record.document_id))
        section_heading = self._coerce_optional_string(metadata.get("section_heading"))
        source_uri = self._coerce_optional_string(metadata.get("source_uri"))
        locator = self._build_locator(metadata)
        citation = Citation(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            source_type=source_type,
            title=title,
            locator=locator,
            snippet=self._build_snippet(record.text),
            source_uri=source_uri,
        )

        vector_candidate = candidate_payload["vector_candidate"]
        lexical_candidate = candidate_payload["lexical_candidate"]
        retrieval_mode = "hybrid" if self._lexical_index is not None else "vector"

        return RetrievalResult(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            source_type=source_type,
            score=float(candidate_payload["score"]),
            rank=rank,
            text=record.text,
            title=title,
            section_heading=section_heading,
            source_uri=source_uri,
            document_metadata=metadata,
            metadata={
                "vector_score": vector_candidate.normalized_score if vector_candidate else 0.0,
                "lexical_score": lexical_candidate.normalized_score if lexical_candidate else 0.0,
                "vector_raw_score": vector_candidate.raw_score if vector_candidate else 0.0,
                "lexical_raw_score": lexical_candidate.raw_score if lexical_candidate else 0.0,
            },
            provenance={
                "collection_name": self._collection_name,
                "retrieval_mode": retrieval_mode,
                "merge_strategy": "weighted_normalized_sum",
                "vector_weight": self._vector_weight,
                "lexical_weight": self._lexical_weight,
                "vector_rank": vector_candidate.rank if vector_candidate else None,
                "lexical_rank": lexical_candidate.rank if lexical_candidate else None,
                "vector_score": vector_candidate.normalized_score if vector_candidate else 0.0,
                "lexical_score": lexical_candidate.normalized_score if lexical_candidate else 0.0,
            },
            citation=citation,
        )

    def _build_locator(self, metadata: dict[str, Any]) -> str:
        """Create a human-readable locator for result citations.

        Why this function exists:
            Retrieval results should explain where the matched chunk came from.
            This helper turns stored provenance metadata into a stable locator
            string that callers can show directly or use for debugging.

        Parameters:
            metadata: Document and chunk metadata carried forward from the
                indexing manifest.

        Returns:
            A human-readable locator string.

        Edge cases handled:
            When structured provenance is missing, the helper falls back to the
            chunk index or a generic label so citation construction never fails.
        """
        if "locator" in metadata and isinstance(metadata["locator"], str):
            return metadata["locator"]

        section_heading = self._coerce_optional_string(metadata.get("section_heading"))
        if section_heading:
            return f"Section: {section_heading}"

        chunk_index = metadata.get("chunk_index")
        if isinstance(chunk_index, int):
            return f"Chunk {chunk_index}"

        return "Chunk"

    def _build_snippet(self, text: str) -> str:
        """Trim chunk text down to a citation-sized snippet.

        Why this function exists:
            Retrieval results should include an evidence preview, but the
            citation payload should remain compact. This helper keeps snippet
            generation consistent and ties the truncation limit to the shared
            settings object.

        Parameters:
            text: Full chunk text selected by retrieval.

        Returns:
            A snippet short enough for citation display.

        Edge cases handled:
            Text shorter than the configured limit is returned unchanged; longer
            text is truncated with an ellipsis so callers can still see that the
            snippet is incomplete.
        """
        limit = self._settings.citation_character_limit
        if len(text) <= limit:
            return text

        return f"{text[: max(0, limit - 3)].rstrip()}..."

    def _coerce_optional_string(self, value: Any) -> str | None:
        """Normalize optional metadata fields into strings or ``None``.

        Why this function exists:
            Manifest metadata may contain missing, empty, or non-string values.
            Result construction only wants usable text fields, so this helper
            isolates the coercion and empty-string handling in one place.

        Parameters:
            value: Metadata value that may or may not contain meaningful text.

        Returns:
            A stripped string when possible, otherwise ``None``.

        Edge cases handled:
            Empty strings and ``None`` are normalized to ``None`` so downstream
            result fields stay clean and predictable.
        """
        if value is None:
            return None

        normalized_value = str(value).strip()
        return normalized_value or None


def build_retrieval_service(
    *,
    collection_name: str = "default",
    embedding_service: EmbeddingService | None = None,
    settings: Settings | None = None,
    enable_hybrid: bool = True,
) -> RetrievalService:
    """Convenience helper for constructing the local retrieval service.

    Why this function exists:
        Some callers only need the standard local retrieval engine and should
        not have to know about all of the underlying manifest and dependency
        wiring. This helper keeps the default path concise while still using
        the fully modular service underneath.

    Parameters:
        collection_name: Logical collection to load from local index storage.
        embedding_service: Optional embedder override.
        settings: Optional settings override for tests or scripts.
        enable_hybrid: Whether lexical retrieval should be included when
            available.

    Returns:
        A configured ``RetrievalService`` for the requested collection.

    Edge cases handled:
        The helper inherits the same manifest-loading failure behavior as
        ``RetrievalService.from_collection`` so missing required artifacts still
        surface clearly.
    """
    return RetrievalService.from_collection(
        collection_name=collection_name,
        embedding_service=embedding_service,
        settings=settings,
        enable_hybrid=enable_hybrid,
    )
