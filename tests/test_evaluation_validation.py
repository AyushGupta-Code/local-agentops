"""Unit tests for deterministic validation and benchmark recording."""

from pathlib import Path

from app.core.config import Settings
from app.evaluation.service import EvaluationService
from app.evaluation.validation import (
    validate_answer_structure,
    validate_citation_presence,
    validate_citation_source_consistency,
    validate_retrieval_evidence,
)
from app.generation.backends import LocalModelService, StaticJSONLocalModelBackend
from app.generation.service import GenerationService
from app.models.schemas import Citation, EvaluationTestCase, GenerationResponse, RetrievalResult


def _build_retrieval_results() -> list[RetrievalResult]:
    """Create grounded retrieval fixtures shared by validation tests.

    Why this function exists:
        The validation rules need realistic retrieval results with citation
        metadata so they can test source consistency and evidence-threshold
        behavior under the same response shapes used by the real pipeline.

    Parameters:
        This helper does not take parameters because the tests use one stable
        evidence set.

    Returns:
        A list of retrieval results suitable for evaluation and validation
        checks.

    Edge cases handled:
        The fixture includes multiple evidence items with different scores so
        the low-evidence rule can be exercised without fabricating separate
        schema shapes.
    """
    return [
        RetrievalResult(
            chunk_id="doc-1:0",
            document_id="doc-1",
            source_type="document",
            score=0.84,
            rank=1,
            text="Reset passwords from the login page using the emailed reset link.",
            title="Password Reset Guide",
            section_heading="Reset Steps",
            source_uri="kb://password-reset",
            document_metadata={"source_uri": "kb://password-reset"},
            metadata={"vector_score": 0.8, "lexical_score": 0.7},
            provenance={"retrieval_mode": "hybrid"},
            citation=Citation(
                chunk_id="doc-1:0",
                document_id="doc-1",
                source_type="document",
                title="Password Reset Guide",
                locator="Section: Reset Steps",
                snippet="Reset passwords from the login page using the emailed reset link.",
                source_uri="kb://password-reset",
            ),
        ),
        RetrievalResult(
            chunk_id="doc-1:1",
            document_id="doc-1",
            source_type="document",
            score=0.33,
            rank=2,
            text="Check the spam folder if the reset email does not arrive.",
            title="Password Reset Guide",
            section_heading="Troubleshooting",
            source_uri="kb://password-reset",
            document_metadata={"source_uri": "kb://password-reset"},
            metadata={"vector_score": 0.3, "lexical_score": 0.3},
            provenance={"retrieval_mode": "hybrid"},
            citation=Citation(
                chunk_id="doc-1:1",
                document_id="doc-1",
                source_type="document",
                title="Password Reset Guide",
                locator="Section: Troubleshooting",
                snippet="Check the spam folder if the reset email does not arrive.",
                source_uri="kb://password-reset",
            ),
        ),
    ]


def _build_generation_response() -> GenerationResponse:
    """Create a grounded answer fixture used by validation tests.

    Why this function exists:
        Several validation rules operate on the final answer payload rather than
        on raw model output. This helper keeps those tests focused on validation
        behavior instead of repeated response construction noise.

    Parameters:
        This helper takes no parameters because one stable grounded response is
        enough for the current validation checks.

    Returns:
        A valid grounded ``GenerationResponse``.

    Edge cases handled:
        The response includes exactly one citation and one supporting evidence
        summary so alignment-based validation rules can test both pass and fail
        cases by making small targeted edits.
    """
    citation = _build_retrieval_results()[0].citation
    return GenerationResponse(
        answer="Use the reset link from the login page.",
        citations=[citation],
        used_chunk_ids=["doc-1:0"],
        supporting_evidence_summary=["Password Reset Guide (Section: Reset Steps): Reset passwords from the login page using the emailed reset link."],
        confidence=0.84,
        confidence_notes=[],
        follow_up_questions=[],
        model_name="mock-model",
    )


def test_validate_citation_presence_fails_when_answer_has_no_citations() -> None:
    """Verify that citation presence validation catches uncited answers."""
    response = GenerationResponse(
        answer="Use the reset link from the login page.",
        citations=[],
        used_chunk_ids=[],
        supporting_evidence_summary=[],
        confidence=None,
        confidence_notes=[],
        follow_up_questions=[],
        model_name="mock-model",
    )

    outcome = validate_citation_presence(response)

    assert outcome.passed is False
    assert outcome.severity == "error"


def test_validate_citation_source_consistency_fails_for_unknown_sources() -> None:
    """Verify that citation-source consistency rejects unknown chunk ids."""
    invalid_citation = Citation(
        chunk_id="outside:1",
        document_id="outside",
        source_type="document",
        title="Outside Source",
        locator="Section: Unknown",
        snippet="Outside evidence.",
        source_uri="kb://outside",
    )
    response = GenerationResponse(
        answer="Use the reset link from the login page.",
        citations=[invalid_citation],
        used_chunk_ids=["outside:1"],
        supporting_evidence_summary=["Outside evidence."],
        confidence=None,
        confidence_notes=[],
        follow_up_questions=[],
        model_name="mock-model",
    )

    outcome = validate_citation_source_consistency(response, _build_retrieval_results())

    assert outcome.passed is False
    assert "outside the retrieved evidence set" in outcome.details


def test_validate_retrieval_evidence_flags_empty_or_low_evidence() -> None:
    """Verify that retrieval validation flags empty and weak evidence sets."""
    empty_outcome = validate_retrieval_evidence([])
    low_score_result = _build_retrieval_results()[1]
    low_outcome = validate_retrieval_evidence([low_score_result], minimum_score=0.5)

    assert empty_outcome.passed is False
    assert empty_outcome.severity == "warning"
    assert low_outcome.passed is False
    assert "below the minimum threshold" in low_outcome.details


def test_validate_answer_structure_flags_misaligned_answer_payloads() -> None:
    """Verify that answer-structure validation catches inconsistent lengths."""
    response = GenerationResponse(
        answer="Use the reset link from the login page.",
        citations=[_build_retrieval_results()[0].citation],
        used_chunk_ids=["doc-1:0", "doc-1:1"],
        supporting_evidence_summary=["Password Reset Guide summary."],
        confidence=None,
        confidence_notes=[],
        follow_up_questions=[],
        model_name="mock-model",
    )

    outcome = validate_answer_structure(response)

    assert outcome.passed is False
    assert "citation_to_chunk_alignment=False" in outcome.details


def test_evaluation_service_runs_case_and_saves_results(tmp_path: Path) -> None:
    """Verify that the benchmark harness records and persists evaluation output."""
    settings = Settings(
        DATA_ROOT=tmp_path,
        RAW_DATA_DIR="raw",
        PROCESSED_DATA_DIR="processed",
        INDEX_DATA_DIR="indexes",
        EVAL_DATA_DIR="eval",
    )
    backend = StaticJSONLocalModelBackend(
        {
            "answer": "Use the reset link from the login page.",
            "cited_chunk_ids": ["doc-1:0"],
            "confidence_notes": [],
            "follow_up_questions": [],
        },
        model_name="benchmark-backend",
    )
    generation_service = GenerationService(
        local_model_service=LocalModelService(backend=backend, settings=settings),
        settings=settings,
    )
    evaluation_service = EvaluationService(
        generation_service=generation_service,
        settings=settings,
    )
    test_case = EvaluationTestCase(
        evaluation_id="case-1",
        query="How do I reset my password?",
        task_type="document_qa",
        retrieved_context=_build_retrieval_results(),
    )

    record = evaluation_service.run_case(dataset_name="smoke", test_case=test_case)
    output_path = evaluation_service.save_results(dataset_name="smoke", records=[record])

    assert record.final_answer == "Use the reset link from the login page."
    assert record.selected_evidence_chunk_ids == ["doc-1:0"]
    assert record.validation_outcomes
    assert record.latency_ms >= 0.0
    assert output_path.exists()
    assert output_path == settings.resolve_eval_data_path("smoke/latest_results.json")
