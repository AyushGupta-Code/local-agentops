"""Deterministic validation rules for retrieval and grounded generation."""

from app.models.schemas import GenerationResponse, RetrievalResult, ValidationOutcome


def validate_citation_presence(response: GenerationResponse) -> ValidationOutcome:
    """Validate that an answer includes explicit citations when it uses evidence.

    Why this function exists:
        A grounded answer should surface the evidence it relied on. This rule
        checks that the generation layer did not return a free-form answer
        without exposing any citations, which would make later auditing and UI
        traceability difficult.

    Parameters:
        response: Final generation response produced by the answer assembly
            layer.

    Returns:
        A ``ValidationOutcome`` describing whether citations are present.

    Limitations:
        This rule only checks for the presence of structured citations. It does
        not prove that the cited evidence actually supports every statement in
        the answer.
    """
    has_citations = bool(response.citations)
    details = (
        "Answer contains structured citations."
        if has_citations
        else "Answer did not include any structured citations."
    )
    return ValidationOutcome(
        rule_name="citation_presence",
        passed=has_citations,
        severity="error",
        details=details,
    )


def validate_citation_source_consistency(
    response: GenerationResponse,
    retrieved_context: list[RetrievalResult],
) -> ValidationOutcome:
    """Validate that answer citations come from the retrieved evidence set.

    Why this function exists:
        Citation integrity matters more than citation count. This rule checks
        that every citation and used chunk id in the final answer resolves back
        to the retrieval results that were actually supplied to generation,
        which guards against citation fabrication or accidental source drift.

    Parameters:
        response: Final generation response whose citations should be checked.
        retrieved_context: Evidence set originally passed to the generator.

    Returns:
        A ``ValidationOutcome`` describing whether citation references stay
        inside the retrieved evidence set.

    Limitations:
        This rule verifies source identity only. It does not compare the answer
        prose semantically against the cited snippets.
    """
    retrieved_chunk_ids = {result.chunk_id for result in retrieved_context}
    unknown_citations = [
        citation.chunk_id
        for citation in response.citations
        if citation.chunk_id not in retrieved_chunk_ids
    ]
    unknown_used_chunks = [
        chunk_id
        for chunk_id in response.used_chunk_ids
        if chunk_id not in retrieved_chunk_ids
    ]
    passed = not unknown_citations and not unknown_used_chunks
    details = (
        "All citations and used chunk ids map back to retrieved evidence."
        if passed
        else (
            "Found citation references outside the retrieved evidence set: "
            f"citations={unknown_citations}, used_chunk_ids={unknown_used_chunks}"
        )
    )
    return ValidationOutcome(
        rule_name="citation_source_consistency",
        passed=passed,
        severity="error",
        details=details,
    )


def validate_retrieval_evidence(
    retrieved_context: list[RetrievalResult],
    *,
    minimum_results: int = 1,
    minimum_score: float = 0.2,
) -> ValidationOutcome:
    """Validate that the retrieval stage returned enough usable evidence.

    Why this function exists:
        Generation quality usually collapses when retrieval returns no evidence
        or only very weak evidence. This rule gives the benchmark harness a
        simple first-pass check for empty or low-evidence retrieval before more
        nuanced metrics are added.

    Parameters:
        retrieved_context: Retrieval hits made available to the generation
            layer.
        minimum_results: Minimum acceptable number of retrieved chunks.
        minimum_score: Minimum acceptable top retrieval score.

    Returns:
        A ``ValidationOutcome`` indicating whether the retrieval evidence is
        non-empty and above a simple relevance threshold.

    Limitations:
        This rule uses only count and score thresholds. It does not determine
        whether the retrieved chunks are actually the best possible evidence for
        the query.
    """
    if len(retrieved_context) < minimum_results:
        return ValidationOutcome(
            rule_name="retrieval_evidence_quality",
            passed=False,
            severity="warning",
            details=(
                f"Retrieved {len(retrieved_context)} chunk(s), which is below the required minimum "
                f"of {minimum_results}."
            ),
        )

    top_score = max(result.score for result in retrieved_context)
    passed = top_score >= minimum_score
    details = (
        f"Retrieved evidence meets the minimum score threshold with top score {top_score:.3f}."
        if passed
        else (
            f"Retrieved evidence is weak: top score {top_score:.3f} is below the "
            f"minimum threshold {minimum_score:.3f}."
        )
    )
    return ValidationOutcome(
        rule_name="retrieval_evidence_quality",
        passed=passed,
        severity="warning",
        details=details,
    )


def validate_answer_structure(response: GenerationResponse) -> ValidationOutcome:
    """Validate that the assembled answer payload has the expected structure.

    Why this function exists:
        The generation layer promises a structured response with answer text,
        evidence summaries, and chunk tracking. This rule checks a few key
        invariants so benchmark records can flag malformed outputs early.

    Parameters:
        response: Final generation response assembled from local model output.

    Returns:
        A ``ValidationOutcome`` describing whether the answer structure looks
        valid for the current first-pass evaluation harness.

    Limitations:
        This rule focuses on simple structural invariants and not answer
        quality. A structurally valid answer can still be factually wrong or
        weakly grounded.
    """
    has_answer = bool(response.answer.strip())
    evidence_alignment = len(response.citations) == len(response.supporting_evidence_summary)
    chunk_alignment = len(response.citations) == len(response.used_chunk_ids)
    passed = has_answer and evidence_alignment and chunk_alignment

    details = (
        "Answer structure is internally consistent."
        if passed
        else (
            "Invalid answer structure detected: "
            f"has_answer={has_answer}, "
            f"citation_to_summary_alignment={evidence_alignment}, "
            f"citation_to_chunk_alignment={chunk_alignment}."
        )
    )
    return ValidationOutcome(
        rule_name="answer_structure",
        passed=passed,
        severity="error",
        details=details,
    )


def run_default_validations(
    *,
    response: GenerationResponse,
    retrieved_context: list[RetrievalResult],
) -> list[ValidationOutcome]:
    """Run the default first-pass validation bundle for one generated answer.

    Why this function exists:
        The evaluation harness needs a predictable set of validation checks for
        every case so result records are easy to compare and audit. This helper
        centralizes that bundle and preserves a stable rule order.

    Parameters:
        response: Final generation response to validate.
        retrieved_context: Retrieval evidence used to produce the response.

    Returns:
        A list of ``ValidationOutcome`` objects in deterministic rule order.

    Limitations:
        The bundle intentionally stays small and auditable. It is not trying to
        score semantic correctness or statistical model quality yet.
    """
    return [
        validate_retrieval_evidence(retrieved_context),
        validate_citation_presence(response),
        validate_citation_source_consistency(response, retrieved_context),
        validate_answer_structure(response),
    ]
