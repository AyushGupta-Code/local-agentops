"""Chunking strategies for different parsed document shapes."""

from app.models.schemas import DocumentChunk, ParsedDocument
from app.chunking.utils import (
    approximate_token_count,
    build_chunk_metadata,
    derive_strategy_name,
    get_section_for_offset,
    split_text_with_overlap,
)


def _build_chunk(
    document: ParsedDocument,
    *,
    chunk_index: int,
    char_start: int,
    char_end: int,
    text: str,
    section_heading: str | None,
    strategy_name: str,
) -> DocumentChunk:
    """Create one deterministic chunk object with preserved provenance.

    Why this function exists:
        All chunking strategies ultimately produce the same ``DocumentChunk``
        schema. This helper keeps chunk ID construction, size estimation, and
        provenance metadata consistent across strategies.

    Parameters:
        document: Parsed document that owns the chunk.
        chunk_index: Zero-based chunk number within the document.
        char_start: Inclusive chunk start offset in the parsed document.
        char_end: Exclusive chunk end offset in the parsed document.
        text: Chunk body text ready for indexing.
        section_heading: Optional section title associated with the chunk.
        strategy_name: Name of the chunking strategy that produced the chunk.

    Returns:
        A fully populated ``DocumentChunk`` ready for the indexing stage.

    Edge cases handled:
        Deterministic IDs are generated from document ID and character range, so
        repeated chunking of unchanged content produces stable chunk IDs.
    """
    metadata = build_chunk_metadata(
        document,
        char_start=char_start,
        char_end=char_end,
        section_heading=section_heading,
    )
    metadata["chunking_strategy"] = strategy_name

    return DocumentChunk(
        chunk_id=f"{document.document_id}:{chunk_index}:{char_start}:{char_end}",
        document_id=document.document_id,
        source_type=document.source_type,
        text=text,
        token_count_estimate=approximate_token_count(text),
        char_start=char_start,
        char_end=char_end,
        chunk_index=chunk_index,
        section_heading=section_heading,
        metadata=metadata,
    )


def chunk_plain_text_document(
    document: ParsedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Chunk plain-text documents using overlapping sliding windows.

    Why this function exists:
        Plain text documents may not expose reliable structural boundaries, so a
        character-window strategy provides a predictable fallback that still
        preserves some surrounding context via overlap.

    Parameters:
        document: Parsed document whose plain text should be split into windows.
        chunk_size: Maximum target chunk size in characters.
        chunk_overlap: Number of trailing characters to repeat into the next
            chunk for boundary continuity.

    Returns:
        A list of deterministic ``DocumentChunk`` objects covering the document
        text in order.

    Edge cases handled:
        Very short documents produce one chunk; empty window artifacts are
        skipped by the splitter helper; overlap is realized as repeated text at
        chunk boundaries rather than separate metadata only.
    """
    windows = split_text_with_overlap(
        document.plain_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[DocumentChunk] = []
    for chunk_index, (char_start, char_end, text) in enumerate(windows):
        owning_section = get_section_for_offset(document.sections, char_start)
        chunks.append(
            _build_chunk(
                document,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                text=text,
                section_heading=owning_section.heading if owning_section else None,
                strategy_name="plain_text",
            )
        )

    return chunks


def chunk_section_based_document(
    document: ParsedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Chunk structured documents by section, then by overlapping windows.

    Why this function exists:
        Markdown and other sectioned documents benefit from respecting heading
        boundaries because section titles are high-value retrieval features.
        This strategy preserves those boundaries first, then subdivides long
        sections with overlap as needed.

    Parameters:
        document: Parsed document whose sections should anchor chunk boundaries.
        chunk_size: Maximum target chunk size in characters within a section.
        chunk_overlap: Number of trailing characters repeated between adjacent
            chunks inside the same section.

    Returns:
        A list of deterministic ``DocumentChunk`` objects ordered by section and
        then by position within each section.

    Edge cases handled:
        Documents with no sections fall back to plain-text chunking; sections
        shorter than the configured size remain intact as one chunk.
    """
    if not document.sections:
        return chunk_plain_text_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    chunks: list[DocumentChunk] = []
    chunk_index = 0
    for section in document.sections:
        windows = split_text_with_overlap(
            section.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for local_start, local_end, text in windows:
            char_start = section.start_offset + local_start
            char_end = section.start_offset + local_end
            chunks.append(
                _build_chunk(
                    document,
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                    text=text,
                    section_heading=section.heading,
                    strategy_name="section_based",
                )
            )
            chunk_index += 1

    return chunks


def chunk_support_ticket_document(
    document: ParsedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Chunk support tickets using section-aware conversational segments.

    Why this function exists:
        Support tickets usually contain short summary fields plus one or more
        conversational turns. Treating those sections as semantically distinct
        helps retrieval keep ticket context intact while still allowing overlap
        inside long conversation bodies.

    Parameters:
        document: Parsed support-ticket document produced by the parsing layer.
        chunk_size: Maximum target chunk size in characters.
        chunk_overlap: Number of trailing characters repeated between adjacent
            chunks inside the same ticket section.

    Returns:
        A list of ``DocumentChunk`` objects optimized for support-ticket
        retrieval and grounded answer generation.

    Edge cases handled:
        Tickets without sections fall back to plain-text chunking; short summary
        sections remain single chunks so metadata is not diluted across windows.
    """
    if not document.sections:
        return chunk_plain_text_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    chunks: list[DocumentChunk] = []
    chunk_index = 0
    for section in document.sections:
        # Keep very small ticket sections intact so important metadata fields
        # like status or ticket ID stay together in one retrieval unit.
        if len(section.text) <= chunk_size:
            chunks.append(
                _build_chunk(
                    document,
                    chunk_index=chunk_index,
                    char_start=section.start_offset,
                    char_end=section.end_offset,
                    text=section.text,
                    section_heading=section.heading,
                    strategy_name="support_ticket",
                )
            )
            chunk_index += 1
            continue

        windows = split_text_with_overlap(
            section.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for local_start, local_end, text in windows:
            char_start = section.start_offset + local_start
            char_end = section.start_offset + local_end
            chunks.append(
                _build_chunk(
                    document,
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                    text=text,
                    section_heading=section.heading,
                    strategy_name="support_ticket",
                )
            )
            chunk_index += 1

    return chunks


def chunk_document_by_strategy(
    document: ParsedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Dispatch a parsed document to the appropriate chunking strategy.

    Why this function exists:
        The service layer should be able to ask for chunks without duplicating
        strategy selection logic. This dispatcher keeps strategy choice
        deterministic and centralized.

    Parameters:
        document: Parsed document to chunk.
        chunk_size: Maximum target chunk size in characters.
        chunk_overlap: Number of trailing characters repeated between adjacent
            chunks generated by the selected strategy.

    Returns:
        A list of deterministic ``DocumentChunk`` objects.

    Edge cases handled:
        Support tickets override section-based chunking because conversation
        semantics matter more than generic document structure for that source
        type.
    """
    strategy_name = derive_strategy_name(document)
    if strategy_name == "support_ticket":
        return chunk_support_ticket_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if strategy_name == "section_based":
        return chunk_section_based_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return chunk_plain_text_document(
        document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
