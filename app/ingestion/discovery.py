"""File discovery helpers for the ingestion pipeline."""

from pathlib import Path


def discover_files(input_directories: list[Path], *, recursive: bool = True) -> list[Path]:
    """Discover concrete files beneath one or more input directories.

    Why this function exists:
        Ingestion begins with filesystem discovery. Keeping that logic separate
        from format loading makes it easier to test directory scanning behavior
        without coupling it to file parsing decisions.

    Parameters:
        input_directories: Directories that should be scanned for candidate
            source files.
        recursive: Whether nested subdirectories should also be scanned.

    Returns:
        A sorted list of file paths discovered under the provided directories.

    Edge cases handled:
        Nonexistent directories are ignored rather than raising immediately, and
        duplicate file paths are removed when overlapping scan roots are used.
    """
    # Use a set so overlapping directories cannot produce duplicate file paths
    # when one scan root is nested beneath another.
    discovered_paths: set[Path] = set()

    # Walk each configured input directory and collect files only, because
    # ingestion should register sources rather than empty directories.
    for input_directory in input_directories:
        if not input_directory.exists() or not input_directory.is_dir():
            continue

        iterator = input_directory.rglob("*") if recursive else input_directory.glob("*")
        for candidate_path in iterator:
            if candidate_path.is_file():
                discovered_paths.add(candidate_path.resolve(strict=False))

    # Sort the result for deterministic ingestion order and stable tests.
    discovered_files = sorted(discovered_paths)

    # Return the ordered file list to the caller.
    return discovered_files
