"""Deterministic query classification for orchestration planning."""

import re
from dataclasses import dataclass

from app.models.schemas import OrchestrationRequest, TaskClassification


@dataclass(frozen=True)
class RoutingRule:
    """One inspectable rule used by the deterministic query router."""

    task_type: str
    name: str
    patterns: tuple[str, ...]
    weight: float


class QueryRouter:
    """Rule-based router that maps user requests to task types."""

    def __init__(self) -> None:
        """Initialize the static routing rules used for task classification.

        Why this function exists:
            The routing layer should stay deterministic and easy to inspect.
            Defining the rules in the constructor keeps the full classifier
            configuration visible in one place and makes later extension a
            straightforward matter of adding rules instead of changing hidden
            model behavior.

        Parameters:
            This initializer does not accept external parameters because the
            current router is intentionally static and deterministic.

        Returns:
            This initializer stores the rule set on the router instance.

        Edge cases handled:
            Using explicit weighted rules allows the router to resolve mixed
            requests predictably even when a query matches more than one task
            family.
        """
        self._rules = self._build_rules()

    def classify(self, request: OrchestrationRequest) -> TaskClassification:
        """Classify one user request into a deterministic orchestration task type.

        Why this function exists:
            The orchestrator needs one transparent task label before it can
            choose downstream components. This method normalizes the user
            request, scores every route using explicit keyword and phrase rules,
            and returns both the winning task and the reasoning artifacts that
            explain why that route was selected.

        Parameters:
            request: Typed orchestration request carrying the raw user message.

        Returns:
            A ``TaskClassification`` containing the selected task type,
            matched rules, alternative task types, and a heuristic confidence.

        Edge cases handled:
            Requests that do not strongly match a specialized route fall back to
            ``document_qa`` because question answering over indexed knowledge is
            the safest default path in this project.
        """
        normalized_request = self._normalize_request(request.user_request)
        task_scores = {task_type: 0.0 for task_type in self._task_types()}
        matched_rules_by_task = {task_type: [] for task_type in self._task_types()}

        for rule in self._rules:
            if self._rule_matches(normalized_request, rule):
                task_scores[rule.task_type] += rule.weight
                matched_rules_by_task[rule.task_type].append(rule.name)

        if all(score == 0.0 for score in task_scores.values()):
            task_scores["document_qa"] = 0.2
            matched_rules_by_task["document_qa"].append("default_question_answering_fallback")

        primary_task_type = max(
            task_scores,
            key=lambda task_type: (
                task_scores[task_type],
                self._task_priority(task_type),
            ),
        )
        primary_score = task_scores[primary_task_type]
        total_score = sum(task_scores.values())
        alternative_task_types = [
            task_type
            for task_type, _ in sorted(
                task_scores.items(),
                key=lambda item: (item[1], self._task_priority(item[0])),
                reverse=True,
            )
            if task_type != primary_task_type and task_scores[task_type] > 0.0
        ]

        # Confidence is a routing transparency aid rather than a learned
        # probability. It rises when many winning rules fire and when the
        # winning route is clearly separated from the alternatives.
        routing_confidence = min(
            1.0,
            0.35 + (primary_score / max(0.5, total_score)),
        )

        return TaskClassification(
            task_type=primary_task_type,
            matched_rules=matched_rules_by_task[primary_task_type],
            alternative_task_types=alternative_task_types,
            routing_confidence=routing_confidence,
        )

    def _build_rules(self) -> list[RoutingRule]:
        """Define the weighted routing rules for all supported task types.

        Why this function exists:
            Query routing needs a single authoritative rule table so the
            orchestration layer remains inspectable and easy to extend. Keeping
            the rule definitions in one helper makes it obvious why each route
            exists and which phrases activate it.

        Parameters:
            This helper accepts no parameters because it returns the router's
            fixed static configuration.

        Returns:
            A list of weighted ``RoutingRule`` objects spanning all supported
            task types.

        Edge cases handled:
            Multiple overlapping patterns are allowed intentionally so a query
            can accumulate evidence for a route instead of depending on one
            brittle exact phrase.
        """
        return [
            RoutingRule("policy_lookup", "policy_keywords", ("policy", "guideline", "compliance", "allowed", "prohibited"), 1.0),
            RoutingRule("policy_lookup", "procedure_keywords", ("procedure", "retention", "security policy", "access policy"), 0.8),
            RoutingRule("ticket_similarity", "ticket_similarity_keywords", ("similar ticket", "similar issue", "related incident", "look like this ticket"), 1.2),
            RoutingRule("ticket_similarity", "incident_language", ("incident", "case like", "ticket like", "prior case"), 0.8),
            RoutingRule("summarization", "summary_keywords", ("summarize", "summary", "brief overview", "condense"), 1.1),
            RoutingRule("summarization", "meeting_or_thread_summary", ("summarize this thread", "summarize this document", "key takeaways"), 0.9),
            RoutingRule("field_extraction", "extraction_keywords", ("extract", "pull out", "find fields", "structured data"), 1.1),
            RoutingRule("field_extraction", "field_names", ("customer id", "ticket id", "priority", "owner", "due date"), 0.9),
            RoutingRule("reply_drafting", "reply_keywords", ("draft a reply", "write a reply", "respond to", "email reply"), 1.2),
            RoutingRule("reply_drafting", "customer_response_language", ("reply to customer", "response draft", "follow-up reply"), 0.9),
            RoutingRule("document_qa", "question_keywords", ("what", "how", "why", "where", "when"), 0.5),
            RoutingRule("document_qa", "knowledge_lookup_keywords", ("find information", "look up", "answer from docs", "knowledge base"), 0.8),
        ]

    def _normalize_request(self, user_request: str) -> str:
        """Normalize raw request text before applying routing rules.

        Why this function exists:
            Routing rules should be written against a stable text form instead
            of dealing with case, punctuation, and whitespace variations in
            every rule. Normalization keeps the rule table compact and the
            classification behavior easier to debug.

        Parameters:
            user_request: Raw user request string submitted for routing.

        Returns:
            A lowercase whitespace-normalized request string.

        Edge cases handled:
            Punctuation is replaced with spaces rather than removed blindly so
            phrase boundaries remain readable for multi-word route patterns.
        """
        collapsed_request = re.sub(r"[^\w\s]", " ", user_request.lower())
        return re.sub(r"\s+", " ", collapsed_request).strip()

    def _rule_matches(self, normalized_request: str, rule: RoutingRule) -> bool:
        """Check whether a routing rule should contribute to the request score.

        Why this function exists:
            Every supported route is driven by explicit lexical evidence.
            Centralizing the match logic makes the router inspectable and gives
            future maintainers one place to change phrase matching semantics.

        Parameters:
            normalized_request: Normalized request text prepared by the router.
            rule: Candidate routing rule being evaluated.

        Returns:
            ``True`` when any pattern in the rule is present in the request.

        Edge cases handled:
            Multi-word phrases and single keywords are handled uniformly by
            simple substring matching because the router is intentionally
            deterministic and not trying to infer hidden intent.
        """
        return any(pattern in normalized_request for pattern in rule.patterns)

    def _task_types(self) -> tuple[str, ...]:
        """Return the supported task types handled by the router.

        Why this function exists:
            The router and orchestrator both need a stable enumeration of task
            types. Keeping the list in one helper avoids subtle drift between
            score initialization, fallback behavior, and tests.

        Parameters:
            This helper takes no parameters because the supported task types are
            fixed by the current deterministic planner.

        Returns:
            A tuple of supported orchestration task types.

        Edge cases handled:
            Returning a tuple instead of a mutable list reduces accidental rule
            mutation during tests or future extension work.
        """
        return (
            "document_qa",
            "policy_lookup",
            "ticket_similarity",
            "summarization",
            "field_extraction",
            "reply_drafting",
        )

    def _task_priority(self, task_type: str) -> int:
        """Provide deterministic tie-breaking priority between task types.

        Why this function exists:
            Some user requests will match more than one route. A fixed priority
            order ensures classification remains stable and inspectable instead
            of depending on dictionary iteration or incidental rule ordering.

        Parameters:
            task_type: Candidate task type whose priority should be returned.

        Returns:
            An integer priority where higher values win ties.

        Edge cases handled:
            Unknown task types receive the lowest priority so future partial
            extensions cannot accidentally outrank the supported routes.
        """
        priorities = {
            "reply_drafting": 6,
            "field_extraction": 5,
            "ticket_similarity": 4,
            "policy_lookup": 3,
            "summarization": 2,
            "document_qa": 1,
        }
        return priorities.get(task_type, 0)
