"""Path helper functions for local storage directories."""

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist.

    Args:
        path: Filesystem path that should exist as a directory.

    Returns:
        The same path object after the directory creation attempt succeeds.
    """
    # Create the directory tree and suppress errors when it already exists.
    path.mkdir(parents=True, exist_ok=True)

    # Return the normalized path so callers can continue chaining operations.
    return path
