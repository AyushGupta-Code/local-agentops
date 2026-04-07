"""Command-line runner for local vector and hybrid retrieval."""

import argparse
import json

from app.models.schemas import RetrievalRequest
from app.retrieval.embeddings import HashEmbeddingService
from app.retrieval.service import build_retrieval_service


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for running retrieval against local manifests.

    Why this function exists:
        Retrieval should be easy to exercise from the terminal without writing
        ad hoc Python. Isolating parser construction keeps the script's command
        contract explicit and easy to extend.

    Parameters:
        This helper does not take explicit parameters because it only builds the
        parser definition.

    Returns:
        A configured ``ArgumentParser`` for the retrieval runner.

    Edge cases handled:
        Defaults are supplied for collection, ranking depth, and hybrid mode so
        the happy path remains a one-command local workflow.
    """
    parser = argparse.ArgumentParser(description="Run local retrieval against an indexed collection.")
    parser.add_argument("--query", required=True, help="Query string to retrieve against the collection.")
    parser.add_argument("--collection", default="default", help="Logical collection name to load.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum number of ranked results to return.")
    parser.add_argument(
        "--vector-only",
        action="store_true",
        help="Disable lexical retrieval and only use vector similarity.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum fused score threshold in the range 0..1.",
    )
    return parser


def main() -> int:
    """Execute retrieval from the command line and print JSON results.

    Why this function exists:
        Developers need a compact way to inspect retrieval behavior while the
        generation layer is still intentionally absent. This script wires the
        request model, retrieval service, and JSON output together into that
        minimal workflow.

    Parameters:
        This function reads command-line arguments from ``sys.argv`` and
        therefore accepts no explicit parameters.

    Returns:
        Process exit code ``0`` on success.

    Edge cases handled:
        Empty result sets are serialized as an empty JSON array so shell users
        can distinguish "no matches" from script failure.
    """
    arguments = build_argument_parser().parse_args()
    retrieval_service = build_retrieval_service(
        collection_name=arguments.collection,
        embedding_service=HashEmbeddingService(),
        enable_hybrid=not arguments.vector_only,
    )
    request = RetrievalRequest(
        query=arguments.query,
        top_k=arguments.top_k,
        min_score=arguments.min_score,
    )
    results = retrieval_service.retrieve(request)

    print(json.dumps([result.model_dump(mode="json") for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
