"""Deterministic orchestration planner driven by query classification."""

from app.models.schemas import (
    OrchestrationPlan,
    OrchestrationRequest,
    OrchestrationStep,
    TaskClassification,
)
from app.orchestration.router import QueryRouter


class OrchestrationService:
    """Plan builder that maps routed tasks onto deterministic pipeline steps."""

    def __init__(self, *, query_router: QueryRouter | None = None) -> None:
        """Store the query router used to classify requests before planning.

        Why this function exists:
            The orchestration layer should remain modular and easy to extend.
            Injecting the router here allows the project to evolve the
            classification logic independently from the downstream plan
            templates without introducing autonomous behavior.

        Parameters:
            query_router: Optional deterministic query router used to classify
                requests before plan construction.

        Returns:
            This initializer stores the router on the service instance.

        Edge cases handled:
            A default router is created automatically so callers can use the
            standard deterministic planner without extra wiring.
        """
        self._query_router = query_router or QueryRouter()

    def build_plan(self, request: OrchestrationRequest) -> OrchestrationPlan:
        """Classify a request and return the matching orchestration plan.

        Why this function exists:
            The caller needs one structured object describing which pipeline
            components should run for the request. This method performs the
            routing step first, then selects a task-specific plan template so
            the downstream path remains deterministic and inspectable.

        Parameters:
            request: Typed user request that should be routed and planned.

        Returns:
            An ``OrchestrationPlan`` describing the exact component sequence for
            the selected task type.

        Edge cases handled:
            The plan is always built from an explicit task classification, so
            even fallback routing still yields a valid deterministic plan rather
            than an ambiguous "agent decides" behavior.
        """
        classification = self._query_router.classify(request)
        return self._plan_for_task(request.user_request, classification)

    def _plan_for_task(
        self,
        user_request: str,
        classification: TaskClassification,
    ) -> OrchestrationPlan:
        """Dispatch to the plan template for the selected task type.

        Why this function exists:
            Each task type intentionally follows a different deterministic
            pipeline path. Centralizing that dispatch keeps the orchestrator
            transparent and makes it straightforward to add new routes without
            turning the planner into a free-form decision engine.

        Parameters:
            user_request: Original user request string.
            classification: Task classification produced by the query router.

        Returns:
            A task-specific ``OrchestrationPlan``.

        Edge cases handled:
            Every supported task type has an explicit plan builder, and unknown
            values fall back to the document-QA path to preserve deterministic
            behavior during partial extensions.
        """
        planners = {
            "document_qa": self._build_document_qa_plan,
            "policy_lookup": self._build_policy_lookup_plan,
            "ticket_similarity": self._build_ticket_similarity_plan,
            "summarization": self._build_summarization_plan,
            "field_extraction": self._build_field_extraction_plan,
            "reply_drafting": self._build_reply_drafting_plan,
        }
        planner = planners.get(classification.task_type, self._build_document_qa_plan)
        return planner(user_request, classification)

    def _build_document_qa_plan(
        self,
        user_request: str,
        classification: TaskClassification,
    ) -> OrchestrationPlan:
        """Build the plan for direct question answering over indexed documents.

        Why this route exists:
            ``document_qa`` is the default knowledge-work path for user
            questions that ask for an answer grounded in stored documents or
            tickets.

        Downstream components called:
            The plan uses ``retrieval`` to gather supporting chunks and then
            hands those chunks to ``generation`` for a grounded answer.

        How this orchestration path differs:
            Unlike the other routes, this path optimizes for a direct answer
            rather than policy-focused filtering, structural extraction, thread
            condensation, or draft composition.

        Parameters:
            user_request: Original user message being planned.
            classification: Router output that selected the ``document_qa``
                path.

        Returns:
            An ``OrchestrationPlan`` for document question answering.

        Edge cases handled:
            The route remains valid as a fallback when no more specialized
            routing evidence is present, which is why it includes general
            retrieval and answer synthesis steps only.
        """
        steps = [
            self._make_step(
                step_id="retrieve_context",
                component="retrieval",
                action="retrieve_ranked_chunks",
                purpose="Gather the most relevant evidence chunks before answer generation.",
                inputs={"query_source": "user_request", "retrieval_mode": "hybrid"},
                outputs=["retrieval_results"],
            ),
            self._make_step(
                step_id="generate_answer",
                component="generation",
                action="compose_grounded_answer",
                purpose="Produce a direct answer using the retrieved evidence with citations.",
                inputs={"query_source": "user_request", "context_source": "retrieval_results"},
                outputs=["answer_response"],
            ),
        ]
        return self._assemble_plan(
            user_request=user_request,
            classification=classification,
            steps=steps,
            completion_artifact="citation_grounded_answer",
            notes=["Generation remains a separate component; this plan only fixes the deterministic path."],
        )

    def _build_policy_lookup_plan(
        self,
        user_request: str,
        classification: TaskClassification,
    ) -> OrchestrationPlan:
        """Build the plan for policy and guideline lookups.

        Why this route exists:
            Policy questions usually need tighter retrieval semantics than
            general document QA because the user is asking for authoritative
            rules, procedures, or compliance guidance.

        Downstream components called:
            The plan uses ``retrieval`` first and then ``generation`` to return
            a concise policy-grounded answer once the relevant policy passages
            have been found.

        How this orchestration path differs:
            Compared with ``document_qa``, this path explicitly annotates the
            retrieval intent as policy-focused so downstream retrieval filters or
            prompts can stay conservative and provenance-heavy.

        Parameters:
            user_request: Original user message being planned.
            classification: Router output that selected the ``policy_lookup``
                path.

        Returns:
            An ``OrchestrationPlan`` for policy lookup.

        Edge cases handled:
            Even when the corpus does not yet expose a dedicated policy filter,
            the plan remains inspectable by recording the policy-focused intent
            in the step inputs instead of silently treating the request as a
            generic question.
        """
        steps = [
            self._make_step(
                step_id="retrieve_policy_evidence",
                component="retrieval",
                action="retrieve_ranked_chunks",
                purpose="Find authoritative policy or procedural passages relevant to the request.",
                inputs={"query_source": "user_request", "retrieval_mode": "hybrid", "intent": "policy_lookup"},
                outputs=["policy_retrieval_results"],
            ),
            self._make_step(
                step_id="generate_policy_answer",
                component="generation",
                action="compose_grounded_answer",
                purpose="Summarize the located policy evidence into a precise answer with citations.",
                inputs={"query_source": "user_request", "context_source": "policy_retrieval_results"},
                outputs=["policy_answer_response"],
            ),
        ]
        return self._assemble_plan(
            user_request=user_request,
            classification=classification,
            steps=steps,
            completion_artifact="policy_grounded_answer",
            notes=["The policy route keeps the same core components as document QA but with stricter intent annotation."],
        )

    def _build_ticket_similarity_plan(
        self,
        user_request: str,
        classification: TaskClassification,
    ) -> OrchestrationPlan:
        """Build the plan for finding similar historical tickets or incidents.

        Why this route exists:
            Some users are not asking for a direct answer. They want prior cases
            that resemble the current ticket so they can compare symptoms,
            resolutions, or escalation patterns.

        Downstream components called:
            This path calls ``retrieval`` only because similarity search is the
            primary artifact; there is no need to invoke generation unless the
            product later adds a separate comparison summary step.

        How this orchestration path differs:
            Unlike answer-oriented paths, this route ends with ranked evidence
            rather than synthesized prose, keeping the workflow deterministic
            and transparent.

        Parameters:
            user_request: Original user message being planned.
            classification: Router output that selected the
                ``ticket_similarity`` path.

        Returns:
            An ``OrchestrationPlan`` for similar-ticket retrieval.

        Edge cases handled:
            The route records a support-ticket retrieval intent explicitly so
            downstream filters can narrow the corpus without requiring the
            router to mutate the user request itself.
        """
        steps = [
            self._make_step(
                step_id="retrieve_similar_tickets",
                component="retrieval",
                action="retrieve_ranked_chunks",
                purpose="Find support-ticket chunks whose symptoms or resolutions resemble the request.",
                inputs={
                    "query_source": "user_request",
                    "retrieval_mode": "hybrid",
                    "intent": "ticket_similarity",
                    "source_types": ["support_ticket"],
                },
                outputs=["similar_ticket_results"],
            ),
        ]
        return self._assemble_plan(
            user_request=user_request,
            classification=classification,
            steps=steps,
            completion_artifact="ranked_similar_tickets",
            notes=["This route deliberately stops at retrieval because the primary output is similar evidence, not a drafted answer."],
        )

    def _build_summarization_plan(
        self,
        user_request: str,
        classification: TaskClassification,
    ) -> OrchestrationPlan:
        """Build the plan for summarizing retrieved content.

        Why this route exists:
            Summarization requests ask the system to condense material rather
            than answer a single factual question. They still need retrieval
            first so the summarizer works from the right evidence subset.

        Downstream components called:
            The plan calls ``retrieval`` to gather the source content and then
            ``generation`` to produce a summary from that retrieved context.

        How this orchestration path differs:
            The output artifact is a summary rather than a question answer, so
            the generation step is framed as condensation of context instead of
            direct response synthesis.

        Parameters:
            user_request: Original user message being planned.
            classification: Router output that selected the ``summarization``
                path.

        Returns:
            An ``OrchestrationPlan`` for summarization.

        Edge cases handled:
            This route remains deterministic by always retrieving first instead
            of allowing a hidden model decision about whether summarization
            should work from raw user text or indexed context.
        """
        steps = [
            self._make_step(
                step_id="retrieve_summary_context",
                component="retrieval",
                action="retrieve_ranked_chunks",
                purpose="Gather the most relevant content that should be condensed into a summary.",
                inputs={"query_source": "user_request", "retrieval_mode": "hybrid", "intent": "summarization"},
                outputs=["summary_context"],
            ),
            self._make_step(
                step_id="generate_summary",
                component="generation",
                action="compose_summary",
                purpose="Condense the retrieved context into a concise structured summary.",
                inputs={"query_source": "user_request", "context_source": "summary_context"},
                outputs=["summary_response"],
            ),
        ]
        return self._assemble_plan(
            user_request=user_request,
            classification=classification,
            steps=steps,
            completion_artifact="grounded_summary",
            notes=["Summarization differs from document QA because the terminal artifact is a condensation of context, not a direct answer."],
        )

    def _build_field_extraction_plan(
        self,
        user_request: str,
        classification: TaskClassification,
    ) -> OrchestrationPlan:
        """Build the plan for structured field extraction.

        Why this route exists:
            Extraction tasks ask for specific fields or structured values
            instead of open-ended prose, which means the orchestrator should
            preserve the expectation of schema-like output.

        Downstream components called:
            The path uses ``retrieval`` to locate the relevant source passages
            and then ``generation`` to format the requested fields from that
            evidence.

        How this orchestration path differs:
            Compared with summarization or QA, this route frames the final step
            as structured extraction rather than narrative generation.

        Parameters:
            user_request: Original user message being planned.
            classification: Router output that selected the
                ``field_extraction`` path.

        Returns:
            An ``OrchestrationPlan`` for field extraction.

        Edge cases handled:
            The route explicitly separates retrieval from extraction so future
            schema validation can be added at a clean extension point without
            changing the router itself.
        """
        steps = [
            self._make_step(
                step_id="retrieve_extraction_context",
                component="retrieval",
                action="retrieve_ranked_chunks",
                purpose="Locate the source passages that contain the requested fields.",
                inputs={"query_source": "user_request", "retrieval_mode": "hybrid", "intent": "field_extraction"},
                outputs=["extraction_context"],
            ),
            self._make_step(
                step_id="extract_fields",
                component="generation",
                action="extract_structured_fields",
                purpose="Return requested fields from the retrieved context in a structured format.",
                inputs={"query_source": "user_request", "context_source": "extraction_context"},
                outputs=["extracted_fields"],
            ),
        ]
        return self._assemble_plan(
            user_request=user_request,
            classification=classification,
            steps=steps,
            completion_artifact="structured_field_output",
            notes=["The extraction route keeps a dedicated terminal action so later schema enforcement can be attached cleanly."],
        )

    def _build_reply_drafting_plan(
        self,
        user_request: str,
        classification: TaskClassification,
    ) -> OrchestrationPlan:
        """Build the plan for drafting a reply grounded in stored knowledge.

        Why this route exists:
            Reply drafting is different from answering a question because the
            final artifact should be a user-facing communication draft rather
            than an internal factual explanation.

        Downstream components called:
            The path first calls ``retrieval`` to gather relevant guidance or
            case context and then calls ``generation`` to draft the reply from
            that evidence.

        How this orchestration path differs:
            The route shares the same core components as QA but changes the
            final action, output artifact, and planning notes to reflect
            communication drafting rather than direct answering.

        Parameters:
            user_request: Original user message being planned.
            classification: Router output that selected the ``reply_drafting``
                path.

        Returns:
            An ``OrchestrationPlan`` for grounded reply drafting.

        Edge cases handled:
            The plan remains deterministic by fixing the two-step retrieval then
            drafting flow instead of allowing free-form multi-hop agent
            behavior.
        """
        steps = [
            self._make_step(
                step_id="retrieve_reply_context",
                component="retrieval",
                action="retrieve_ranked_chunks",
                purpose="Gather policy, ticket, or document evidence that should inform the reply.",
                inputs={"query_source": "user_request", "retrieval_mode": "hybrid", "intent": "reply_drafting"},
                outputs=["reply_context"],
            ),
            self._make_step(
                step_id="draft_reply",
                component="generation",
                action="draft_grounded_reply",
                purpose="Produce a user-facing reply draft grounded in the retrieved evidence.",
                inputs={"query_source": "user_request", "context_source": "reply_context"},
                outputs=["reply_draft"],
            ),
        ]
        return self._assemble_plan(
            user_request=user_request,
            classification=classification,
            steps=steps,
            completion_artifact="grounded_reply_draft",
            notes=["This route is deterministic drafting, not an autonomous agent conversation loop."],
        )

    def _make_step(
        self,
        *,
        step_id: str,
        component: str,
        action: str,
        purpose: str,
        inputs: dict[str, object],
        outputs: list[str],
    ) -> OrchestrationStep:
        """Construct one orchestration step in a compact reusable form.

        Why this function exists:
            Each task-specific planner builds similar step objects. A small
            helper keeps those plan builders readable while preserving the full
            structured step contract in every route.

        Parameters:
            step_id: Stable identifier for the step within the plan.
            component: Pipeline component selected for execution.
            action: Concrete action the component should perform.
            purpose: Why this step exists in the task flow.
            inputs: Structured step inputs recorded by the planner.
            outputs: Named outputs expected from the step.

        Returns:
            An ``OrchestrationStep`` ready to insert into a plan.

        Edge cases handled:
            Inputs are copied into the Pydantic model so future callers do not
            accidentally mutate previously assembled plans.
        """
        return OrchestrationStep(
            step_id=step_id,
            component=component,
            action=action,
            purpose=purpose,
            inputs=dict(inputs),
            outputs=list(outputs),
        )

    def _assemble_plan(
        self,
        *,
        user_request: str,
        classification: TaskClassification,
        steps: list[OrchestrationStep],
        completion_artifact: str,
        notes: list[str],
    ) -> OrchestrationPlan:
        """Assemble the final public orchestration plan object.

        Why this function exists:
            All task planners ultimately return the same public plan shape. This
            helper centralizes that assembly so the per-task methods can focus
            on how their downstream path differs rather than duplicating common
            plan boilerplate.

        Parameters:
            user_request: Original request being orchestrated.
            classification: Router decision used to select the task path.
            steps: Ordered plan steps selected for the task.
            completion_artifact: Description of the final artifact the task
                should produce.
            notes: Additional orchestration notes for the caller.

        Returns:
            A complete ``OrchestrationPlan``.

        Edge cases handled:
            The ordered component list is derived directly from the steps so the
            summary cannot drift out of sync with the actual plan body.
        """
        components_to_call = [step.component for step in steps]
        return OrchestrationPlan(
            user_request=user_request,
            classification=classification,
            components_to_call=components_to_call,
            steps=steps,
            completion_artifact=completion_artifact,
            notes=notes,
        )


def build_orchestration_service() -> OrchestrationService:
    """Create the default deterministic orchestration service.

    Why this function exists:
        Most callers only need the standard router and planner wiring. This
        helper exposes that default path without forcing every caller to know
        how the orchestration package is assembled internally.

    Parameters:
        This helper takes no parameters because it returns the project's
        default deterministic planner configuration.

    Returns:
        A ready-to-use ``OrchestrationService`` instance.

    Edge cases handled:
        The function always returns a fully wired deterministic planner, which
        prevents partial or ad hoc orchestration setup in callers.
    """
    return OrchestrationService()
