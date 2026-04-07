"""Format-specific parsers that convert raw sources into normalized documents."""

import json
import re
from pathlib import Path
from typing import Any

from app.models.schemas import ParsedDocument, ParsedSection, RawDocument
from app.parsing.errors import ParsingError
from app.parsing.normalization import normalize_whitespace
from app.parsing.pdf import PdfTextExtractor


def _build_provenance_metadata(document: RawDocument) -> dict[str, Any]:
    """Build baseline provenance metadata for parsed documents.

    Why this function exists:
        Every parser should preserve the same core provenance fields so later
        retrieval and citation stages can trace parsed output back to its
        source. Centralizing that metadata avoids format-specific drift.

    Parameters:
        document: Raw source record currently being parsed.

    Returns:
        A metadata dictionary seeded with source provenance from the raw record.

    Edge cases handled:
        Missing optional provenance fields remain ``None`` instead of being
        invented or omitted inconsistently across parsers.
    """
    # Copy the existing raw-document metadata so ingestion provenance survives
    # into the parsed representation.
    metadata = dict(document.metadata)

    # Attach parser-level provenance keys used by later retrieval and citation
    # logic regardless of file format.
    metadata["source_file"] = metadata.get("source_path") or document.source_uri
    metadata["row_number"] = metadata.get("record_index")
    metadata["parser_mime_type"] = document.mime_type

    # Return the baseline parsed-document metadata to the caller.
    return metadata


def _section_heading_to_text(heading: str) -> str:
    """Convert a Markdown-style heading into a clean section label.

    Why this function exists:
        Markdown heading lines often include leading ``#`` characters and extra
        spaces. Normalizing them here keeps section labels readable and stable.

    Parameters:
        heading: Raw heading line extracted from Markdown content.

    Returns:
        A clean heading string suitable for ``ParsedSection.heading``.

    Edge cases handled:
        Empty or malformed heading lines fall back to the stripped original text
        so parsing does not fail just because a heading is unconventional.
    """
    # Remove leading Markdown heading markers and surrounding whitespace so the
    # resulting section label is user-friendly.
    normalized_heading = re.sub(r"^#+\s*", "", heading).strip()

    # Return the cleaned heading, or the stripped original if removal would
    # leave the section unnamed.
    return normalized_heading or heading.strip()


def _build_section(heading: str, text: str, order: int, start_offset: int) -> ParsedSection:
    """Create one parsed section with computed offsets.

    Why this function exists:
        Multiple parsers need to emit ``ParsedSection`` objects, and they all
        need the same offset calculation strategy. This helper keeps section
        construction consistent.

    Parameters:
        heading: Human-readable section label.
        text: Section body text after normalization.
        order: Zero-based section order within the parsed document.
        start_offset: Character offset where the section starts in the full
            normalized document text.

    Returns:
        A populated ``ParsedSection`` with start and end offsets.

    Edge cases handled:
        Empty text is not allowed and will fail through the schema layer,
        ensuring sections always point to real content.
    """
    # Compute the end offset from the normalized section text length so later
    # chunking can map sections back into the document body.
    end_offset = start_offset + len(text)

    # Return the structured section to the caller.
    return ParsedSection(
        heading=heading,
        text=text,
        order=order,
        start_offset=start_offset,
        end_offset=end_offset,
    )


def parse_text_document(document: RawDocument) -> ParsedDocument:
    """Parse TXT or generic plain-text sources into one normalized document.

    Why this function exists:
        Plain-text sources do not need structural parsing beyond whitespace
        cleanup and provenance preservation. This function keeps that simple
        path separate from more structured formats like Markdown and PDF.

    Parameters:
        document: Raw document whose content is already available as plain text.

    Returns:
        A normalized ``ParsedDocument`` with one synthetic section covering the
        full text.

    Edge cases handled:
        Whitespace-only content raises ``ParsingError`` because later retrieval
        stages cannot do anything useful with an empty parsed body.
    """
    normalized_text = normalize_whitespace(document.content)
    if not normalized_text:
        raise ParsingError(f"Text document {document.document_id} did not contain readable text")

    sections = [_build_section("Document Body", normalized_text, 0, 0)]
    metadata = _build_provenance_metadata(document)
    metadata["section_headings"] = [section.heading for section in sections]

    return ParsedDocument(
        document_id=document.document_id,
        source_type=document.source_type,
        title=document.title,
        plain_text=normalized_text,
        summary=None,
        language=document.language,
        sections=sections,
        source_uri=document.source_uri,
        metadata=metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def parse_markdown_document(document: RawDocument) -> ParsedDocument:
    """Parse Markdown into normalized text and heading-based sections.

    Why this function exists:
        Markdown has lightweight structure that is valuable to preserve for
        later chunking and citation formatting. This parser extracts headings
        without attempting full Markdown rendering.

    Parameters:
        document: Raw Markdown-style document whose content should be split on
            heading boundaries.

    Returns:
        A normalized ``ParsedDocument`` containing one section per heading block
        when headings exist, or a single fallback section otherwise.

    Edge cases handled:
        Documents without headings fall back to one synthetic section; malformed
        Markdown is still parsed as plain text after normalization.
    """
    normalized_markdown = normalize_whitespace(document.content)
    if not normalized_markdown:
        raise ParsingError(f"Markdown document {document.document_id} did not contain readable text")

    heading_pattern = re.compile(r"^(#{1,6}\s+.+)$", flags=re.MULTILINE)
    matches = list(heading_pattern.finditer(normalized_markdown))

    sections: list[ParsedSection] = []
    if not matches:
        sections.append(_build_section("Document Body", normalized_markdown, 0, 0))
    else:
        plain_text_parts: list[str] = []
        current_offset = 0
        section_index = 0

        # Preserve any introductory Markdown content that appears before the
        # first heading because that text can still be useful retrieval context.
        if matches[0].start() > 0:
            intro_text = normalize_whitespace(normalized_markdown[: matches[0].start()])
            if intro_text:
                sections.append(_build_section("Introduction", intro_text, section_index, current_offset))
                plain_text_parts.append(intro_text)
                current_offset += len(intro_text)
                section_index += 1
                plain_text_parts.append("\n\n")
                current_offset += 2

        for index, match in enumerate(matches):
            section_start = match.start()
            section_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_markdown)
            raw_block = normalized_markdown[section_start:section_end].strip()
            lines = raw_block.split("\n")
            heading = _section_heading_to_text(lines[0])
            body_text = normalize_whitespace("\n".join(lines[1:])) or heading
            section_text = f"{heading}\n{body_text}" if body_text != heading else heading
            sections.append(_build_section(heading, section_text, section_index, current_offset))
            plain_text_parts.append(section_text)
            current_offset += len(section_text)
            section_index += 1
            if index < len(matches) - 1:
                plain_text_parts.append("\n\n")
                current_offset += 2

        normalized_markdown = "".join(plain_text_parts)

    metadata = _build_provenance_metadata(document)
    metadata["section_headings"] = [section.heading for section in sections]

    return ParsedDocument(
        document_id=document.document_id,
        source_type=document.source_type,
        title=document.title,
        plain_text=normalized_markdown,
        summary=None,
        language=document.language,
        sections=sections,
        source_uri=document.source_uri,
        metadata=metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def parse_pdf_document(document: RawDocument, *, extractor: PdfTextExtractor) -> ParsedDocument:
    """Parse a PDF document into normalized per-page sections.

    Why this function exists:
        PDFs need a dedicated parser because provenance by page matters and text
        extraction depends on a backend extractor. Passing the extractor in
        keeps the design open for future OCR implementations.

    Parameters:
        document: Raw PDF document whose original source path should be read.
        extractor: Pluggable page-level PDF text extractor implementation.

    Returns:
        A ``ParsedDocument`` whose sections correspond to normalized PDF pages.

    Edge cases handled:
        Missing source paths, unreadable PDFs, and pages with no extractable
        text raise ``ParsingError`` so operators can see and fix the problem.
    """
    if not document.source_uri:
        raise ParsingError(f"PDF document {document.document_id} is missing a source path")

    source_path = Path(document.source_uri)
    page_texts = extractor.extract_pages(source_path)

    sections: list[ParsedSection] = []
    plain_text_parts: list[str] = []
    current_offset = 0
    page_provenance: list[dict[str, Any]] = []

    for page_index, page_text in enumerate(page_texts):
        heading = f"Page {page_index + 1}"
        sections.append(_build_section(heading, page_text, page_index, current_offset))
        plain_text_parts.append(page_text)
        page_provenance.append(
            {
                "page_number": page_index + 1,
                "start_offset": current_offset,
                "end_offset": current_offset + len(page_text),
                "section_heading": heading,
            }
        )
        current_offset += len(page_text)
        if page_index < len(page_texts) - 1:
            plain_text_parts.append("\n\n")
            current_offset += 2

    plain_text = "".join(plain_text_parts)
    metadata = _build_provenance_metadata(document)
    metadata["page_map"] = page_provenance
    metadata["section_headings"] = [section.heading for section in sections]

    return ParsedDocument(
        document_id=document.document_id,
        source_type=document.source_type,
        title=document.title,
        plain_text=plain_text,
        summary=None,
        language=document.language,
        sections=sections,
        source_uri=document.source_uri,
        metadata=metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def parse_ticket_document(document: RawDocument) -> ParsedDocument:
    """Parse a CSV- or JSON-derived support-ticket record into normalized text.

    Why this function exists:
        Tabular support-ticket exports already arrive as one logical ticket per
        raw record after ingestion. This parser focuses on clean normalization
        and provenance retention rather than re-parsing the original dataset.

    Parameters:
        document: Raw support-ticket record created from a CSV or JSON row.

    Returns:
        A normalized ``ParsedDocument`` containing structured sections for the
        ticket summary and body.

    Edge cases handled:
        Missing row metadata is tolerated, but whitespace-only ticket content
        still raises ``ParsingError`` because it would not support retrieval.
    """
    normalized_body = normalize_whitespace(document.content)
    if not normalized_body:
        raise ParsingError(f"Support ticket {document.document_id} did not contain readable text")

    header_lines = [f"Title: {document.title}"]
    if document.external_id:
        header_lines.append(f"Ticket ID: {document.external_id}")
    if document.ticket_status:
        header_lines.append(f"Status: {document.ticket_status}")
    if document.priority:
        header_lines.append(f"Priority: {document.priority}")
    if document.author_name:
        header_lines.append(f"Requester: {document.author_name}")

    ticket_header = normalize_whitespace("\n".join(header_lines))
    sections: list[ParsedSection] = []
    current_offset = 0

    sections.append(_build_section("Ticket Summary", ticket_header, 0, current_offset))
    current_offset += len(ticket_header) + 2
    sections.append(_build_section("Ticket Body", normalized_body, 1, current_offset))

    plain_text = f"{ticket_header}\n\n{normalized_body}"
    metadata = _build_provenance_metadata(document)
    metadata["row_number"] = document.metadata.get("record_index")
    metadata["section_headings"] = [section.heading for section in sections]

    return ParsedDocument(
        document_id=document.document_id,
        source_type=document.source_type,
        title=document.title,
        plain_text=plain_text,
        summary=None,
        language=document.language,
        sections=sections,
        source_uri=document.source_uri,
        metadata=metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def parse_json_document(document: RawDocument) -> ParsedDocument:
    """Parse a non-ticket JSON document into normalized plain text.

    Why this function exists:
        Some JSON sources are document-like records rather than ticket rows.
        This parser normalizes their content while preserving file provenance.

    Parameters:
        document: Raw JSON-backed document that should be normalized.

    Returns:
        A ``ParsedDocument`` with one normalized section covering the document.

    Edge cases handled:
        JSON-like strings that cannot be decoded are treated as plain text, and
        whitespace-only content still raises ``ParsingError``.
    """
    normalized_text = normalize_whitespace(document.content)
    if not normalized_text:
        raise ParsingError(f"JSON document {document.document_id} did not contain readable text")

    try:
        # Reformat JSON content when it is valid JSON so nested structures
        # become more readable to later retrieval and generation stages.
        decoded_payload = json.loads(normalized_text)
        normalized_text = normalize_whitespace(json.dumps(decoded_payload, indent=2, ensure_ascii=True))
    except json.JSONDecodeError:
        # Leave non-JSON string content untouched because some ingestion paths
        # already stringify row data or mixed payloads before parsing.
        pass

    sections = [_build_section("Document Body", normalized_text, 0, 0)]
    metadata = _build_provenance_metadata(document)
    metadata["section_headings"] = [section.heading for section in sections]

    return ParsedDocument(
        document_id=document.document_id,
        source_type=document.source_type,
        title=document.title,
        plain_text=normalized_text,
        summary=None,
        language=document.language,
        sections=sections,
        source_uri=document.source_uri,
        metadata=metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )
