"""Text normalization helpers for the parsing layer."""

import re


BROKEN_ARTIFACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\u0000"), ""),
    (re.compile(r"\u00ad"), ""),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
)


def normalize_newlines(text: str) -> str:
    """Convert mixed newline styles into Unix newlines.

    Why this function exists:
        Source files often arrive with Windows or old-Mac newline styles. The
        rest of the pipeline should not need to care about those transport
        differences, so parsing normalizes them immediately.

    Parameters:
        text: Raw text extracted from a source file or ticket record.

    Returns:
        The input text with all newline variants converted to ``\\n``.

    Edge cases handled:
        Empty strings remain empty, and mixed newline styles are normalized in a
        single pass without changing other characters.
    """
    # Replace carriage-return based newline variants with the single newline
    # form expected by the rest of the pipeline.
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Return the newline-normalized string to the caller.
    return normalized_text


def remove_broken_text_artifacts(text: str) -> str:
    """Remove a small set of obvious low-quality text artifacts.

    Why this function exists:
        Early extraction often brings along null bytes, soft hyphens, or noisy
        spacing. Removing the most obvious artifacts improves normalized output
        without pretending to solve full layout reconstruction.

    Parameters:
        text: Raw or partially normalized text content that may contain known
            extraction artifacts.

    Returns:
        A cleaned text string with the configured artifact patterns removed or
        reduced.

    Edge cases handled:
        Empty input stays empty, and only a conservative set of patterns is
        modified so legitimate punctuation or formatting is not aggressively
        stripped away.
    """
    # Apply each conservative cleanup rule in order so obvious extraction noise
    # is removed before whitespace is trimmed.
    cleaned_text = text
    for pattern, replacement in BROKEN_ARTIFACT_PATTERNS:
        cleaned_text = pattern.sub(replacement, cleaned_text)

    # Return the cleaned string for any further normalization steps.
    return cleaned_text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries.

    Why this function exists:
        Retrieval and chunking work better when text has predictable spacing,
        but over-normalizing can destroy useful paragraph structure. This helper
        strikes a middle ground by preserving blank-line boundaries.

    Parameters:
        text: Raw or partially cleaned text that may contain inconsistent
            spacing, blank lines, or trailing spaces.

    Returns:
        Text with normalized line endings, collapsed intra-line whitespace, and
        consistent paragraph spacing.

    Edge cases handled:
        Empty or whitespace-only input becomes an empty string, and repeated
        blank lines are collapsed to at most one empty paragraph boundary.
    """
    # Standardize line endings first so later paragraph splitting behaves
    # consistently across platforms.
    newline_normalized_text = normalize_newlines(text)

    # Remove obvious extraction artifacts before trimming individual lines.
    artifact_cleaned_text = remove_broken_text_artifacts(newline_normalized_text)

    # Trim each line independently so paragraph structure survives while noisy
    # leading and trailing spaces are removed.
    normalized_lines = [line.strip() for line in artifact_cleaned_text.split("\n")]

    # Rejoin the cleaned lines and collapse multiple blank paragraphs into one.
    normalized_text = "\n".join(normalized_lines).strip()
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)

    # Return the normalized text to the caller.
    return normalized_text
