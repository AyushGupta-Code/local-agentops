"""Parsing-specific exceptions used by the text extraction layer."""


class ParsingError(Exception):
    """Raised when a raw source cannot be parsed into a normalized document."""
