"""Utility helpers for chunk sizing, overlap, and provenance calculations."""

import re
from typing import Any

from app.models.schemas import ParsedDocument, ParsedSection


def approximate_word_count(text: str) -> int:
    """Estimate the number of words in a chunk of text.

    Why this function exists:
        Early chunking decisions and retrieval budgeting often need a simple
        size estimate without pulling in a model-specific tokenizer. Word count
        is a cheap approximation that remains useful for debugging and tests.

    Parameters:
        text: Normalized text whose approximate word count should be measured.

    Returns:
        The number of whitespace-delimited word-like tokens found in the text.

    Edge cases handled:
        Empty or whitespace-only input returns ``0`` instead of producing a
        misleading minimum count.
    """
    # Count simple word-like spans so chunk metadata can expose a lightweight
    # size estimate without depending on the embedding model tokenizer.
    words = re.findall(r"\b\w+\b", text)

    # Return the resulting word count to the caller.
    return len(words)


def approximate_token_count(text: str) -> int:
    """Estimate token count from text length and word count.

    Why this function exists:
        The indexing and generation stages eventually need token-oriented size
        estimates, but exact tokenization varies by model. This helper provides
        a deterministic approximation suitable for chunk metadata today.

    Parameters:
        text: Normalized text whose approximate token count should be measured.

    Returns:
        A conservative token estimate based on both character length and word
        count.

    Edge cases handled:
        Empty input returns ``0``; very short inputs still return a small but
        non-negative estimate rather than exploding due to division logic.
    """
    if not text.strip():
        return 0

    # Combine a character-based heuristic with a word count heuristic so the
    # estimate behaves reasonably across short and long chunks.
    word_estimate = approximate_word_count(text)
    character_estimate = max(1, len(text) // 4)

    # Return the larger estimate to avoid under-budgeting prompt context later.
    return max(word_estimate, character_estimate)


def split_text_with_overlap(text: str, *, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    """Split plain text into overlapping character windows on word boundaries.

    Why this function exists:
        Plain text documents often lack explicit structure, so the chunker needs
        a fallback sliding-window strategy. This helper implements that strategy
        while trying to avoid cutting words in half when possible.

    Parameters:
        text: Normalized text to split into retrieval-ready windows.
        chunk_size: Maximum target number of characters per chunk window.
        chunk_overlap: Number of trailing characters from one chunk that should
            be repeated at the beginning of the next chunk.

    Returns:
        A list of tuples containing ``(char_start, char_end, chunk_text)`` for
        each generated chunk.

    Edge cases handled:
        Empty text returns an empty list; overlap larger than or equal to the
        chunk size is rejected by the caller configuration; very short text
        returns one chunk covering the entire string.
    """
    if not text:
        return []

    windows: list[tuple[int, int, str]] = []
    text_length = len(text)
    start_index = 0

    while start_index < text_length:
        candidate_end = min(start_index + chunk_size, text_length)

        # Try to end on whitespace when the current window does not already hit
        # the document boundary. This keeps chunks more readable for retrieval.
        if candidate_end < text_length:
            whitespace_index = text.rfind(" ", start_index, candidate_end)
            if whitespace_index > start_index:
                candidate_end = whitespace_index

        chunk_text = text[start_index:candidate_end].strip()

        # Skip pathological empty windows that can happen when long whitespace
        # runs interact with boundary adjustments.
        if not chunk_text:
            start_index = max(candidate_end, start_index + 1)
            continue

        chunk_char_end = start_index + len(chunk_text)
        windows.append((start_index, chunk_char_end, chunk_text))

        if chunk_char_end >= text_length:
            break

        # Move the next window backward by the configured overlap so adjacent
        # chunks share context near their boundaries.
        next_start = max(0, chunk_char_end - chunk_overlap)

        # Avoid getting stuck when trimming or overlap would keep the next
        # window at the same start position.
        if next_start <= start_index:
            next_start = chunk_char_end
        else:
            # Advance to the next non-space character so overlap does not start
            # on a run of whitespace.
            while next_start < text_length and text[next_start].isspace():
                next_start += 1

        start_index = next_start

    # Return the generated window list to the caller.
    return windows


def get_section_for_offset(sections: list[ParsedSection], char_start: int) -> ParsedSection | None:
    """Find the parsed section that owns a given character offset.

    Why this function exists:
        Chunk provenance should include the nearest section heading whenever the
        parsed document exposes section structure. This helper maps chunk
        offsets back onto those sections.

    Parameters:
        sections: Parsed document sections with validated character offsets.
        char_start: Inclusive character offset of the chunk start.

    Returns:
        The matching ``ParsedSection`` when one contains the offset, otherwise
        ``None``.

    Edge cases handled:
        Documents without sections return ``None``; offsets that fall between
        sections also return ``None`` instead of guessing.
    """
    for section in sections:
        if section.start_offset <= char_start < section.end_offset:
            return section
    return None


def get_page_numbers_for_range(document: ParsedDocument, char_start: int, char_end: int) -> list[int]:
    """Resolve PDF-style page numbers that overlap a chunk character span.

    Why this function exists:
        PDF parsing records page provenance in ``ParsedDocument.metadata``.
        Chunking needs to carry that information forward so citations can later
        point back to the correct pages.

    Parameters:
        document: Parsed document whose metadata may include a page map.
        char_start: Inclusive chunk start offset within the parsed document.
        char_end: Exclusive chunk end offset within the parsed document.

    Returns:
        A sorted list of page numbers touched by the chunk range.

    Edge cases handled:
        Non-PDF documents or documents without a page map return an empty list;
        chunks spanning multiple pages include each touched page once.
    """
    page_map = document.metadata.get("page_map")
    if not isinstance(page_map, list):
        return []

    page_numbers: list[int] = []
    for page_entry in page_map:
        if not isinstance(page_entry, dict):
            continue
        page_start = page_entry.get("start_offset")
        page_end = page_entry.get("end_offset")
        page_number = page_entry.get("page_number")
        if not isinstance(page_start, int) or not isinstance(page_end, int) or not isinstance(page_number, int):
            continue
        overlaps = char_start < page_end and char_end > page_start
        if overlaps and page_number not in page_numbers:
            page_numbers.append(page_number)

    return page_numbers


def build_chunk_metadata(
    document: ParsedDocument,
    *,
    char_start: int,
    char_end: int,
    section_heading: str | None,
) -> dict[str, Any]:
    """Build provenance metadata for one retrieval chunk.

    Why this function exists:
        Chunk objects need to carry enough provenance to support indexing,
        debugging, and later citation rendering. Centralizing that metadata
        creation keeps each chunking strategy consistent.

    Parameters:
        document: Parsed document being chunked.
        char_start: Inclusive chunk start offset within the document.
        char_end: Exclusive chunk end offset within the document.
        section_heading: Optional section heading associated with the chunk.

    Returns:
        A metadata dictionary containing source file and range provenance.

    Edge cases handled:
        Documents without page or row provenance simply store empty or ``None``
        values for those fields instead of fabricating data.
    """
    metadata = {
        "source_file": document.metadata.get("source_file") or document.source_uri,
        "page_numbers": get_page_numbers_for_range(document, char_start, char_end),
        "row_range": None,
        "section_title": section_heading,
        "char_range": [char_start, char_end],
    }

    row_number = document.metadata.get("row_number")
    if isinstance(row_number, int):
        metadata["row_range"] = [row_number, row_number]

    return metadata


def derive_strategy_name(document: ParsedDocument) -> str:
    """Infer the appropriate chunking strategy for a parsed document.

    Why this function exists:
        Different document shapes need different chunking tradeoffs. This helper
        keeps the service dispatch logic explicit and easy to test.

    Parameters:
        document: Parsed document whose source type and sections determine the
            best chunking strategy.

    Returns:
        One of ``plain_text``, ``section_based``, or ``support_ticket``.

    Edge cases handled:
        Documents with sections but generic headings still fall back to
        ``section_based`` because the explicit structure is still more useful
        than a flat sliding window.
    """
    if document.source_type == "support_ticket":
        return "support_ticket"
    if document.sections:
        return "section_based"
    return "plain_text"
