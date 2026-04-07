"""Embedding abstractions and local implementations for retrieval."""

import math
import re
from hashlib import blake2b
from typing import Protocol


class EmbeddingService(Protocol):
    """Protocol describing the embedding operations the retriever needs."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many chunk texts for vector index construction.

        Why this function exists:
            Retrieval needs to embed every indexed chunk when the local vector
            index is built. Using a protocol keeps that concern abstract so the
            retrieval engine can work with deterministic test embedders, local
            model-backed embedders, or future remote providers without changing
            the service logic.

        Parameters:
            texts: Chunk texts that should be converted into vector space.

        Returns:
            A list of dense vectors in the same order as the input texts.

        Edge cases handled:
            An empty input list should produce an empty embedding list so index
            construction remains a predictable no-op.
        """

    def embed_query(self, query: str) -> list[float]:
        """Embed one query string for vector retrieval.

        Why this function exists:
            Query embedding is the first scoring step in vector retrieval. The
            retrieval service keeps it behind a small interface so score
            calculation remains independent from the concrete embedding model.

        Parameters:
            query: User query string that should be projected into vector
                space.

        Returns:
            One dense query vector.

        Edge cases handled:
            Implementations should still return a stable vector for unusual
            inputs so the retrieval service can fail predictably instead of
            special-casing every query shape.
        """


class HashEmbeddingService:
    """Deterministic token-hash embedder for local development and tests."""

    def __init__(self, dimension: int = 32) -> None:
        """Configure a fixed-size embedding space for hashed bag-of-words vectors.

        Why this function exists:
            The project needs a simple embedder that works offline for tests and
            CLI experimentation. A hashed token vector is cheap, deterministic,
            and sufficient to exercise the retrieval flow before a production
            embedding model is introduced.

        Parameters:
            dimension: Size of the dense embedding vector produced for each
                text.

        Returns:
            This initializer stores the embedding dimension on the instance.

        Edge cases handled:
            Very small dimensions are rejected because they make collisions so
            extreme that vector scores become misleading even for toy datasets.
        """
        if dimension < 8:
            raise ValueError("dimension must be at least 8 for stable hashed embeddings")

        self._dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into normalized hashed token vectors.

        Why this function exists:
            The vector index computes chunk embeddings in one batch before any
            query runs. Keeping the batch path explicit makes the retrieval flow
            clear and avoids repeated per-text initialization work.

        Parameters:
            texts: Chunk texts to embed for the in-memory vector index.

        Returns:
            A list of L2-normalized dense vectors aligned with ``texts``.

        Edge cases handled:
            Empty input returns an empty list so callers can build an empty
            index without special handling.
        """
        return [self._embed_text(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        """Embed one query using the same hashing logic used for indexed texts.

        Why this function exists:
            Vector scores are only meaningful when the query and the stored
            chunks live in the same embedding space. This method makes that
            symmetry explicit in the retrieval flow.

        Parameters:
            query: User query whose semantic signal should be compared with
                chunk vectors.

        Returns:
            One L2-normalized dense query vector.

        Edge cases handled:
            Queries that tokenize to nothing still return a stable all-zero
            vector, which allows the retriever to report empty or low-scoring
            results instead of failing.
        """
        return self._embed_text(query)

    def _embed_text(self, text: str) -> list[float]:
        """Convert one text string into a normalized hashed token vector.

        Why this function exists:
            Both chunk indexing and query-time embedding share the same
            tokenization and hashing logic. Centralizing that logic guarantees
            score consistency between the stored vectors and the query vector.

        Parameters:
            text: Raw text to convert into vector space.

        Returns:
            One dense vector whose magnitude is normalized for cosine scoring.

        Edge cases handled:
            Repeated tokens increase the corresponding dimensions just as they
            would in a bag-of-words model, while tokenless input produces a
            stable zero vector that later cosine scoring can handle safely.
        """
        # Tokenization is intentionally simple because this embedder exists to
        # exercise retrieval flow and score handling, not to maximize recall.
        vector = [0.0] * self._dimension
        for token in self._tokenize(text):
            vector[self._hash_token(token)] += 1.0

        # Normalize once so cosine scoring later reflects direction rather than
        # raw token count, which keeps long chunks from dominating by length.
        return self._normalize_vector(vector)

    def _tokenize(self, text: str) -> list[str]:
        """Split text into lowercase word tokens for embedding.

        Why this function exists:
            The hashed embedding model only needs a lightweight token stream.
            Isolating tokenization makes the retrieval flow easier to inspect
            and lets tests reason about why certain chunks score well.

        Parameters:
            text: Raw text to tokenize.

        Returns:
            A lowercase token list.

        Edge cases handled:
            Punctuation-only strings yield an empty token list, which the
            embedding path turns into a zero vector instead of an exception.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def _hash_token(self, token: str) -> int:
        """Map one token deterministically into the configured vector space.

        Why this function exists:
            The offline embedder needs a stable and reproducible way to project
            arbitrary tokens into a fixed number of dimensions. Token hashing
            provides that while keeping the implementation compact.

        Parameters:
            token: One normalized token from the input text.

        Returns:
            The integer vector position that should be incremented for the
            token.

        Edge cases handled:
            The hash is taken modulo the embedding dimension so every token
            maps to a valid slot even though collisions are expected.
        """
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big") % self._dimension

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """Normalize a dense vector for cosine similarity scoring.

        Why this function exists:
            Retrieval score handling depends on vector magnitudes being
            comparable. L2-normalizing here ensures the later cosine similarity
            step compares direction only, which keeps score interpretation
            stable across short and long texts.

        Parameters:
            vector: Dense vector that may or may not already be normalized.

        Returns:
            A normalized copy of ``vector`` or an all-zero vector when the
            magnitude is zero.

        Edge cases handled:
            Zero-magnitude vectors are returned unchanged because dividing by
            zero would fail and there is no directional information to recover.
        """
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return vector

        return [value / magnitude for value in vector]
