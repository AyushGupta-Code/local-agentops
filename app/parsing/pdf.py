"""Pluggable PDF parsing helpers for the text extraction layer."""

from pathlib import Path
from typing import Protocol

from app.parsing.errors import ParsingError
from app.parsing.normalization import normalize_whitespace

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - dependency availability is environment-specific.
    PdfReader = None


class PdfTextExtractor(Protocol):
    """Protocol for pluggable PDF text extraction backends."""

    def extract_pages(self, path: Path) -> list[str]:
        """Extract one text string per page from a PDF source."""


class PypdfTextExtractor:
    """PDF text extractor backed by ``pypdf`` for non-OCR documents."""

    def extract_pages(self, path: Path) -> list[str]:
        """Extract normalized text for each PDF page using ``pypdf``.

        Why this function exists:
            The parsing layer needs one default PDF backend today, while still
            leaving room for future OCR or alternative extraction engines. This
            method provides the non-OCR baseline implementation.

        Parameters:
            path: Filesystem path to the PDF source file.

        Returns:
            A list containing one normalized text string per page in document
            order.

        Edge cases handled:
            Missing ``pypdf`` raises a clear ``ParsingError``; corrupt PDFs and
            pages with no extractable text also raise explicit parse failures
            instead of silently returning misleading empty output.
        """
        # Fail explicitly when the optional PDF dependency is unavailable so the
        # operator knows why PDF parsing cannot proceed.
        if PdfReader is None:
            raise ParsingError("PDF parsing requires the 'pypdf' dependency to be installed")

        try:
            # Open the PDF with pypdf so text can be extracted one page at a
            # time and provenance can be preserved.
            reader = PdfReader(str(path))
        except Exception as error:  # pragma: no cover - backend-specific failures vary.
            raise ParsingError(f"Unable to open PDF file {path.name}: {error}") from error

        extracted_pages: list[str] = []
        for page_index, page in enumerate(reader.pages):
            try:
                # Extract raw text from the current page. Later OCR work can
                # plug in here when scanned PDFs need image-based extraction.
                page_text = page.extract_text() or ""
            except Exception as error:  # pragma: no cover - backend-specific failures vary.
                raise ParsingError(
                    f"Unable to extract text from page {page_index + 1} of {path.name}: {error}"
                ) from error

            normalized_page_text = normalize_whitespace(page_text)
            if not normalized_page_text:
                raise ParsingError(
                    f"PDF page {page_index + 1} in {path.name} did not yield readable text"
                )
            extracted_pages.append(normalized_page_text)

        if not extracted_pages:
            raise ParsingError(f"PDF file {path.name} did not contain any pages")

        # Return the per-page normalized text list to the caller.
        return extracted_pages
