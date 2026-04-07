"""Small benchmark runner for deterministic first-pass evaluation."""

import json

from app.evaluation.service import EvaluationService
from app.generation.backends import LocalModelService, StaticJSONLocalModelBackend
from app.generation.service import GenerationService
from app.models.schemas import Citation, EvaluationTestCase, RetrievalResult


def build_demo_cases() -> list[EvaluationTestCase]:
    """Create a compact deterministic benchmark dataset for local evaluation.

    Why this function exists:
        The benchmark runner should demonstrate the evaluation harness without
        requiring a live retrieval pipeline or local model server. This helper
        builds a small in-memory dataset with explicit evidence and citations so
        the benchmark output remains auditable.

    Parameters:
        This helper takes no parameters because the script is intended as a
        fixed runnable example.

    Returns:
        A list of deterministic evaluation cases.

    Edge cases handled:
        Each case carries retrieval results directly so the benchmark can focus
        on generation latency, validation outcomes, and result persistence
        instead of depending on external services.
    """
    retrieval_result = RetrievalResult(
        chunk_id="doc-1:0",
        document_id="doc-1",
        source_type="document",
        score=0.88,
        rank=1,
        text="Reset your password from the login page using the link sent by email.",
        title="Password Reset Guide",
        section_heading="Reset Steps",
        source_uri="kb://password-reset",
        document_metadata={"source_uri": "kb://password-reset"},
        metadata={"vector_score": 0.85, "lexical_score": 0.8},
        provenance={"retrieval_mode": "hybrid"},
        citation=Citation(
            chunk_id="doc-1:0",
            document_id="doc-1",
            source_type="document",
            title="Password Reset Guide",
            locator="Section: Reset Steps",
            snippet="Reset your password from the login page using the link sent by email.",
            source_uri="kb://password-reset",
        ),
    )
    return [
        EvaluationTestCase(
            evaluation_id="benchmark-1",
            query="How do I reset my password?",
            task_type="document_qa",
            retrieved_context=[retrieval_result],
        )
    ]


def main() -> int:
    """Run a deterministic benchmark and print the saved report path and contents.

    Why this function exists:
        Developers need a minimal end-to-end example showing how benchmark
        cases, generation, validation, and local JSON persistence fit together.
        This script provides that path without introducing a larger benchmark
        framework.

    Parameters:
        This function does not take command-line arguments because the goal is
        to keep the first-pass benchmark example simple and auditable.

    Returns:
        Process exit code ``0`` on success.

    Edge cases handled:
        The script uses a static local backend so it remains deterministic while
        still exercising the same evaluation interfaces used by real backends.
    """
    backend = StaticJSONLocalModelBackend(
        {
            "answer": "Use the reset link on the login page.",
            "cited_chunk_ids": ["doc-1:0"],
            "confidence_notes": [],
            "follow_up_questions": [],
        },
        model_name="benchmark-static-backend",
    )
    generation_service = GenerationService(
        local_model_service=LocalModelService(backend=backend),
    )
    evaluation_service = EvaluationService(generation_service=generation_service)
    cases = build_demo_cases()
    records = evaluation_service.run_benchmark(
        dataset_name="demo-benchmark",
        test_cases=cases,
        save_results=False,
    )
    output_path = evaluation_service.save_results(
        dataset_name="demo-benchmark",
        records=records,
    )

    print(f"Saved benchmark results to: {output_path}")
    print(json.dumps([record.model_dump(mode='json') for record in records], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
