"""Sample script showing one end-to-end grounded generation call."""

import json

from app.generation.backends import LocalModelService, StaticJSONLocalModelBackend
from app.generation.service import GenerationService
from app.models.schemas import Citation, GenerationRequest, RetrievalResult


def build_demo_retrieval_results() -> list[RetrievalResult]:
    """Create a small retrieval fixture batch for the demo generation script.

    Why this function exists:
        The generation demo should be runnable without requiring a live
        retrieval engine or local model server. This helper provides a compact
        evidence set with explicit citation metadata so the sample can exercise
        prompt assembly and answer assembly end to end.

    Parameters:
        This helper does not take parameters because the demo uses one fixed
        evidence set.

    Returns:
        A list of retrieval results that can ground the sample answer.

    Edge cases handled:
        Each result includes concrete citation and provenance metadata so the
        demo reflects the same explicit source-tracking path used in the real
        generation service.
    """
    return [
        RetrievalResult(
            chunk_id="doc-1:0",
            document_id="doc-1",
            source_type="document",
            score=0.92,
            rank=1,
            text="Reset your password from the login page using the reset link sent by email.",
            title="Password Reset Guide",
            section_heading="Reset Steps",
            source_uri="kb://password-reset",
            document_metadata={"source_uri": "kb://password-reset"},
            metadata={"vector_score": 0.91, "lexical_score": 0.82},
            provenance={"retrieval_mode": "hybrid"},
            citation=Citation(
                chunk_id="doc-1:0",
                document_id="doc-1",
                source_type="document",
                title="Password Reset Guide",
                locator="Section: Reset Steps",
                snippet="Reset your password from the login page using the reset link sent by email.",
                source_uri="kb://password-reset",
            ),
        )
    ]


def main() -> int:
    """Run one deterministic grounded generation example from prompt to answer.

    Why this function exists:
        Developers need a concrete example of how to call the local generation
        layer without first wiring a live local model provider. This script
        shows the full request shape, backend abstraction, and structured
        response payload in one place.

    Parameters:
        This function reads no command-line arguments because the goal is to
        provide a minimal runnable example.

    Returns:
        Process exit code ``0`` on success.

    Edge cases handled:
        The script uses a static JSON backend so it remains deterministic and
        still exercises the strict citation validation path.
    """
    backend = StaticJSONLocalModelBackend(
        {
            "answer": "Use the reset link on the login page. The guidance says the link is sent by email.",
            "cited_chunk_ids": ["doc-1:0"],
            "confidence_notes": [],
            "follow_up_questions": [],
        },
        model_name="demo-static-backend",
    )
    generation_service = GenerationService(
        local_model_service=LocalModelService(backend=backend),
    )
    request = GenerationRequest(
        query="How do I reset my password?",
        task_type="document_qa",
        retrieved_context=build_demo_retrieval_results(),
        output_instructions=["Answer in two sentences or fewer."],
    )
    response = generation_service.answer(request)

    print(json.dumps(response.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
