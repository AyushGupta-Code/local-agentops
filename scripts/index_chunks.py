"""Small runner for preparing local index manifests from stored chunks."""

import argparse

from app.indexing.service import IndexingService
from app.storage.service import LocalFileStorageService


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the indexing runner.

    Why this function exists:
        The runner script should be lightweight but still explicit about its
        inputs. Isolating parser creation keeps the command-line contract easy
        to extend later.

    Parameters:
        This helper does not accept parameters because it simply constructs the
        parser object.

    Returns:
        A configured ``ArgumentParser`` for the indexing runner script.

    Edge cases handled:
        Default values are supplied for collection name and lexical preparation
        so the local-first happy path requires minimal flags.
    """
    parser = argparse.ArgumentParser(description="Prepare local vector and lexical index manifests.")
    parser.add_argument("--collection", default="default", help="Logical collection name for manifest output.")
    parser.add_argument(
        "--skip-lexical",
        action="store_true",
        help="Skip lexical index preparation and only build vector manifests.",
    )
    return parser


def main() -> int:
    """Run the local indexing workflow over persisted chunks.

    Why this function exists:
        Developers need a simple entry point for preparing index manifests from
        already persisted chunks. This runner keeps that workflow scriptable
        without coupling it to API endpoints.

    Parameters:
        This function reads command-line arguments from ``sys.argv`` through the
        argument parser and therefore accepts no explicit parameters.

    Returns:
        Process exit code ``0`` on success.

    Edge cases handled:
        Empty chunk stores still produce manifests, which makes no-op runs
        explicit and predictable during local development.
    """
    parser = build_argument_parser()
    arguments = parser.parse_args()

    storage = LocalFileStorageService()
    chunks = storage.load_all_chunks()
    indexing_service = IndexingService(storage=storage)
    result = indexing_service.prepare_indexes(
        chunks,
        collection_name=arguments.collection,
        include_lexical=not arguments.skip_lexical,
    )

    print(f"Prepared {result.chunk_count} chunks")
    print(f"Vector manifest: {result.vector_manifest_path}")
    if result.lexical_manifest_path is not None:
        print(f"Lexical manifest: {result.lexical_manifest_path}")
    print(f"Metadata manifest: {result.metadata_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
