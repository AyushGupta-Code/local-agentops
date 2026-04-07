"""Local vector and lexical retrieval backends built from persisted manifests."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.retrieval.embeddings import EmbeddingService


@dataclass(frozen=True)
class IndexedChunkRecord:
    """Internal representation of one chunk loaded from an index manifest."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalCandidate:
    """Internal ranked candidate emitted by vector or lexical retrieval."""

    record: IndexedChunkRecord
    raw_score: float
    normalized_score: float
    rank: int


def _matches_filters(
    record: IndexedChunkRecord,
    *,
    source_types: list[str],
    tag_filters: list[str],
    include_archived: bool,
) -> bool:
    """Apply shared retrieval filters before any backend-specific scoring.

    Why this function exists:
        Vector and lexical retrieval should score the same candidate pool for a
        given request. Sharing the filter logic avoids subtle ranking drift
        where each backend might otherwise score a different set of records.

    Parameters:
        record: Indexed chunk candidate under consideration.
        source_types: Allowed source families or an empty list for no source
            filter.
        tag_filters: Required lowercase tags or an empty list for no tag
            filter.
        include_archived: Whether archived candidates are eligible.

    Returns:
        ``True`` when the record should remain in the candidate pool.

    Edge cases handled:
        Missing metadata fields default to permissive behavior except for the
        archive flag, where missing metadata is treated as "not archived".
    """
    if source_types:
        source_type = str(record.metadata.get("source_type", "document"))
        if source_type not in source_types:
            return False

    if tag_filters:
        record_tags = {
            str(tag).lower()
            for tag in record.metadata.get("tags", [])
            if isinstance(tag, str)
        }
        if not set(tag_filters).issubset(record_tags):
            return False

    if not include_archived and bool(record.metadata.get("archived", False)):
        return False

    return True


class ManifestVectorIndex:
    """In-memory vector index built from the local vector manifest."""

    def __init__(self, records: list[IndexedChunkRecord], embedding_service: EmbeddingService) -> None:
        """Embed manifest records and prepare them for cosine similarity search.

        Why this function exists:
            The persisted vector manifest currently stores chunk text and
            metadata rather than precomputed dense vectors. This initializer
            performs the embedding step once so query execution can focus on the
            retrieval flow instead of repeatedly embedding every chunk.

        Parameters:
            records: Chunk records loaded from the vector manifest.
            embedding_service: Swappable embedding implementation used for both
                chunk and query embeddings.

        Returns:
            This initializer stores the indexed records and their embeddings on
            the instance.

        Edge cases handled:
            Empty records produce an empty embedding table, which allows the
            retrieval service to return no results cleanly instead of failing.
        """
        self._records = records
        self._embeddings = embedding_service.embed_texts([record.text for record in records])

    @classmethod
    def from_manifest_path(
        cls, manifest_path: Path, embedding_service: EmbeddingService
    ) -> "ManifestVectorIndex":
        """Load vector records from disk and build the in-memory vector index.

        Why this function exists:
            Retrieval runners and tests should not duplicate manifest-loading
            boilerplate. This constructor keeps the data-loading step explicit
            while still hiding the JSON parsing details from the service layer.

        Parameters:
            manifest_path: Filesystem path of the persisted vector-input
                manifest.
            embedding_service: Embedding implementation used to create chunk
                vectors.

        Returns:
            A ready-to-query ``ManifestVectorIndex`` instance.

        Edge cases handled:
            Missing or invalid files surface immediately through standard IO or
            JSON exceptions because retrieval cannot continue with ambiguous
            index state.
        """
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [cls._parse_record(item) for item in payload.get("records", [])]
        return cls(records, embedding_service)

    @staticmethod
    def _parse_record(payload: dict[str, Any]) -> IndexedChunkRecord:
        """Normalize one manifest payload into the internal chunk record type.

        Why this function exists:
            The retrieval engine needs a compact internal representation with a
            predictable field set. Parsing the raw JSON payload once keeps later
            scoring and merge code free from repeated dictionary lookups.

        Parameters:
            payload: Raw JSON object read from the persisted manifest.

        Returns:
            An ``IndexedChunkRecord`` containing chunk identity, text, and
            metadata.

        Edge cases handled:
            Missing metadata falls back to an empty dictionary so filtering and
            result construction can still proceed for minimal manifests.
        """
        return IndexedChunkRecord(
            chunk_id=str(payload["chunk_id"]),
            document_id=str(payload["document_id"]),
            text=str(payload["text"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        source_types: list[str],
        tag_filters: list[str],
        include_archived: bool,
    ) -> dict[str, RetrievalCandidate]:
        """Score all filtered chunks with cosine similarity and keep the best hits.

        Why this function exists:
            Vector retrieval starts after the query has been embedded. This
            method applies the candidate filters, computes cosine similarity for
            every surviving chunk, normalizes the scores into the retriever's
            ``0..1`` range, and returns only the top-ranked candidates for
            hybrid merging.

        Parameters:
            query_embedding: Dense query vector already produced by the
                embedding service.
            top_k: Maximum number of vector candidates to keep.
            source_types: Optional source-type filters applied before scoring.
            tag_filters: Optional tag filters applied before scoring.
            include_archived: Whether records marked archived should remain
                eligible.

        Returns:
            A mapping from ``chunk_id`` to vector retrieval candidates.

        Edge cases handled:
            Empty indexes, empty filtered candidate pools, and zero vectors all
            lead to an empty or low-score result set rather than raising during
            similarity calculation.
        """
        scored_candidates: list[tuple[IndexedChunkRecord, float]] = []
        for record, embedding in zip(self._records, self._embeddings, strict=True):
            if not _matches_filters(
                record,
                source_types=source_types,
                tag_filters=tag_filters,
                include_archived=include_archived,
            ):
                continue

            # Cosine similarity is the raw vector score. It is later shifted
            # into the retriever's ``0..1`` range so vector and lexical scores
            # can be merged without mixing incompatible scales.
            raw_score = self._cosine_similarity(query_embedding, embedding)
            normalized_score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
            scored_candidates.append((record, normalized_score))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        top_candidates = scored_candidates[:top_k]

        return {
            record.chunk_id: RetrievalCandidate(
                record=record,
                raw_score=score,
                normalized_score=score,
                rank=rank,
            )
            for rank, (record, score) in enumerate(top_candidates, start=1)
        }

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        """Compute cosine similarity for already normalized dense vectors.

        Why this function exists:
            Vector score handling should remain explicit. Even though the
            embedding service normalizes vectors, this helper keeps the scoring
            formula visible and isolated for future debugging or replacement.

        Parameters:
            left: Query vector.
            right: Indexed chunk vector.

        Returns:
            Cosine similarity in the theoretical ``-1..1`` range.

        Edge cases handled:
            Empty vectors or mismatched dimensions return ``0.0`` because the
            vectors cannot be compared meaningfully and retrieval should fail
            soft rather than crash mid-query.
        """
        if not left or not right or len(left) != len(right):
            return 0.0

        return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


class ManifestLexicalIndex:
    """In-memory BM25 lexical index built from the local lexical manifest."""

    def __init__(self, records: list[IndexedChunkRecord]) -> None:
        """Build tokenized lexical state for BM25 candidate scoring.

        Why this function exists:
            Hybrid retrieval needs a lexical signal alongside vector similarity.
            This initializer tokenizes the indexed chunks once and constructs a
            BM25 model so query execution only has to score the current query.

        Parameters:
            records: Chunk records loaded from the lexical manifest.

        Returns:
            This initializer stores the lexical records and BM25 state on the
            instance.

        Edge cases handled:
            An empty record list produces a ``None`` BM25 model so later query
            execution can return no lexical candidates gracefully.
        """
        self._records = records
        self._tokenized_corpus = [self._tokenize(record.text) for record in records]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    @classmethod
    def from_manifest_path(cls, manifest_path: Path) -> "ManifestLexicalIndex":
        """Load lexical records from disk and create the BM25-backed index.

        Why this function exists:
            The retrieval service should be able to assemble itself from the
            persisted indexing artifacts. This constructor hides the lexical
            manifest loading details while keeping the retrieval pipeline
            modular.

        Parameters:
            manifest_path: Filesystem path of the persisted lexical-input
                manifest.

        Returns:
            A ready-to-query ``ManifestLexicalIndex`` instance.

        Edge cases handled:
            Invalid or missing manifest files propagate clear exceptions because
            hybrid retrieval cannot build a lexical scorer from incomplete
            state.
        """
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [ManifestVectorIndex._parse_record(item) for item in payload.get("records", [])]
        return cls(records)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        source_types: list[str],
        tag_filters: list[str],
        include_archived: bool,
    ) -> dict[str, RetrievalCandidate]:
        """Score lexical candidates with BM25 and normalize the result range.

        Why this function exists:
            BM25 raw scores are unbounded and therefore cannot be fused directly
            with vector scores. This method executes lexical retrieval,
            normalizes the retained candidates into the ``0..1`` range, and
            returns them in a shape suitable for candidate merging.

        Parameters:
            query: User query string used for lexical matching.
            top_k: Maximum number of lexical candidates to keep.
            source_types: Optional source-type filters applied before ranking.
            tag_filters: Optional tag filters applied before ranking.
            include_archived: Whether archived candidates remain eligible.

        Returns:
            A mapping from ``chunk_id`` to lexical candidates.

        Edge cases handled:
            Empty lexical indexes, tokenless queries, and all-nonpositive BM25
            scores return empty or zero-normalized candidate sets instead of
            crashing the hybrid merge step.
        """
        if self._bm25 is None:
            return {}

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        raw_scores = list(self._bm25.get_scores(query_tokens))
        filtered_candidates: list[tuple[IndexedChunkRecord, float]] = []
        for record, raw_score in zip(self._records, raw_scores, strict=True):
            if not _matches_filters(
                record,
                source_types=source_types,
                tag_filters=tag_filters,
                include_archived=include_archived,
            ):
                continue

            filtered_candidates.append((record, raw_score))

        filtered_candidates.sort(key=lambda item: item[1], reverse=True)
        top_candidates = filtered_candidates[:top_k]
        max_positive_score = max((score for _, score in top_candidates if score > 0.0), default=0.0)

        candidate_mapping: dict[str, RetrievalCandidate] = {}
        for rank, (record, raw_score) in enumerate(top_candidates, start=1):
            # BM25 scores are relative to the current corpus and query, so the
            # merge step needs explicit normalization. Dividing by the best
            # positive lexical score yields a simple ``0..1`` scale.
            normalized_score = raw_score / max_positive_score if max_positive_score > 0.0 else 0.0
            candidate_mapping[record.chunk_id] = RetrievalCandidate(
                record=record,
                raw_score=raw_score,
                normalized_score=max(0.0, min(1.0, normalized_score)),
                rank=rank,
            )

        return candidate_mapping

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing and querying.

        Why this function exists:
            Lexical retrieval should use the same tokenization path for both the
            indexed corpus and the incoming query. Centralizing it keeps BM25
            score behavior understandable during tests and debugging.

        Parameters:
            text: Raw chunk or query text to tokenize.

        Returns:
            Lowercase lexical tokens.

        Edge cases handled:
            Punctuation-only inputs produce an empty token list, which the
            search path treats as a no-match condition instead of an error.
        """
        return re.findall(r"\b\w+\b", text.lower())
