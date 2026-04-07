"""Unit tests for deterministic routing and orchestration planning."""

from app.models.schemas import OrchestrationRequest
from app.orchestration.router import QueryRouter
from app.orchestration.service import OrchestrationService


def test_query_router_classifies_policy_lookup_requests() -> None:
    """Verify that policy-focused requests route to the policy lookup task."""
    router = QueryRouter()

    # Route a compliance-style request because the router should classify it
    # using explicit policy and allowed/prohibited language.
    classification = router.classify(
        OrchestrationRequest(user_request="What is the password retention policy and is reuse allowed?")
    )

    # Assert that the policy route wins and that the matched rule list explains
    # the decision transparently.
    assert classification.task_type == "policy_lookup"
    assert "policy_keywords" in classification.matched_rules
    assert classification.routing_confidence > 0.5


def test_query_router_classifies_ticket_similarity_requests() -> None:
    """Verify that prior-case similarity requests route to ticket similarity."""
    router = QueryRouter()

    # Route a prior-ticket lookup because this request should emphasize similar
    # issues rather than direct answer generation.
    classification = router.classify(
        OrchestrationRequest(user_request="Find a similar ticket for this browser cache sign-in incident.")
    )

    assert classification.task_type == "ticket_similarity"
    assert "ticket_similarity_keywords" in classification.matched_rules


def test_orchestration_service_builds_document_qa_plan() -> None:
    """Verify that general knowledge questions map to retrieval plus generation."""
    orchestration_service = OrchestrationService()

    # Plan a general factual question because the default route should be
    # document QA with retrieval followed by answer generation.
    plan = orchestration_service.build_plan(
        OrchestrationRequest(user_request="How do I reset my password from the docs?")
    )

    assert plan.classification.task_type == "document_qa"
    assert plan.components_to_call == ["retrieval", "generation"]
    assert plan.steps[0].action == "retrieve_ranked_chunks"
    assert plan.steps[1].action == "compose_grounded_answer"
    assert plan.completion_artifact == "citation_grounded_answer"


def test_orchestration_service_builds_summarization_plan() -> None:
    """Verify that summary requests follow the retrieval then summarization path."""
    orchestration_service = OrchestrationService()

    # Plan a summary request because this route should differ from QA by ending
    # in summary generation instead of direct answering.
    plan = orchestration_service.build_plan(
        OrchestrationRequest(user_request="Summarize this document and give me the key takeaways.")
    )

    assert plan.classification.task_type == "summarization"
    assert plan.steps[1].action == "compose_summary"
    assert "Summarization differs from document QA" in plan.notes[0]


def test_orchestration_service_builds_field_extraction_plan() -> None:
    """Verify that extraction requests use the structured extraction route."""
    orchestration_service = OrchestrationService()

    # Plan a field extraction request because this route should preserve the
    # expectation of structured output rather than open-ended prose.
    plan = orchestration_service.build_plan(
        OrchestrationRequest(
            user_request="Extract the customer id, priority, and owner from the matching ticket."
        )
    )

    assert plan.classification.task_type == "field_extraction"
    assert plan.steps[1].action == "extract_structured_fields"
    assert plan.completion_artifact == "structured_field_output"


def test_orchestration_service_builds_reply_drafting_plan() -> None:
    """Verify that reply drafting requests produce a grounded drafting plan."""
    orchestration_service = OrchestrationService()

    # Plan a reply request because the orchestrator should treat this as a
    # communication task rather than a direct answer task.
    plan = orchestration_service.build_plan(
        OrchestrationRequest(
            user_request="Draft a reply to the customer explaining the billing delay."
        )
    )

    assert plan.classification.task_type == "reply_drafting"
    assert plan.steps[1].action == "draft_grounded_reply"
    assert plan.components_to_call == ["retrieval", "generation"]


def test_orchestration_service_builds_retrieval_only_similarity_plan() -> None:
    """Verify that ticket similarity planning stops at retrieval."""
    orchestration_service = OrchestrationService()

    # Plan a similar-ticket request because this route should end with ranked
    # evidence rather than a synthesized answer.
    plan = orchestration_service.build_plan(
        OrchestrationRequest(
            user_request="Look for a related incident or similar ticket for this outage."
        )
    )

    assert plan.classification.task_type == "ticket_similarity"
    assert plan.components_to_call == ["retrieval"]
    assert plan.steps[0].inputs["source_types"] == ["support_ticket"]
