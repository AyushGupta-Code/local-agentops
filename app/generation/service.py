"""Local generation service with explicit prompt and citation handling."""

import json
from typing import Any

from app.core.config import Settings, get_settings
from app.generation.backends import LocalModelService
from app.generation.prompting import PromptAssemblyService
from app.models.schemas import Citation, GenerationRequest, GenerationResponse, RetrievalResult


class AnswerAssemblyService:
    """Validate model output and assemble the final grounded answer response."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        """Store shared settings used during answer assembly.

        Why this function exists:
            Answer assembly needs access to shared limits and metadata defaults,
            but it should stay independent from prompt assembly and backend
            invocation. Keeping that logic in its own service makes citation
            validation and response shaping directly testable.

        Parameters:
            settings: Optional validated settings object.

        Returns:
            This initializer stores the settings on the service instance.

        Edge cases handled:
            A default settings object is created automatically so tests and
            scripts can use the standard assembly behavior without extra setup.
        """
        self._settings = settings or get_settings()

    def assemble_answer(
        self,
        *,
        request: GenerationRequest,
        model_output: str,
        model_name: str,
    ) -> GenerationResponse:
        """Build a grounded generation response from raw model output.

        Why this function exists:
            Local model backends return raw text, but callers need a structured
            answer with verified citations and explicit evidence summaries. This
            method parses the model output, validates every cited chunk id
            against the retrieved evidence, preserves the original citation
            objects from retrieval, and assembles the final public response.

        Parameters:
            request: Original generation request containing the retrieved
                evidence set.
            model_output: Raw backend response text, expected to be JSON.
            model_name: Human-readable model/backend name used for response
                metadata.

        Returns:
            A ``GenerationResponse`` with grounded citations and evidence
            summaries.

        Edge cases handled:
            Invalid JSON, missing required fields, or unknown cited chunk ids
            cause explicit failures rather than silent fallback behavior because
            this layer must never invent or guess citations on behalf of the
            model.
        """
        payload = self._parse_model_output(model_output)
        evidence_by_chunk_id = {
            result.chunk_id: result
            for result in request.retrieved_context
        }
        cited_chunk_ids = self._normalize_cited_chunk_ids(payload.get("cited_chunk_ids", []))
        citations = self._resolve_citations(cited_chunk_ids, evidence_by_chunk_id)
        supporting_evidence_summary = self._build_supporting_evidence_summary(
            cited_chunk_ids,
            evidence_by_chunk_id,
        )
        confidence_notes = self._normalize_string_list(payload.get("confidence_notes", []))
        confidence = self._derive_confidence(
            cited_chunk_ids=cited_chunk_ids,
            evidence_by_chunk_id=evidence_by_chunk_id,
            confidence_notes=confidence_notes,
        )

        return GenerationResponse(
            answer=self._require_non_empty_string(payload.get("answer"), field_name="answer"),
            citations=citations,
            used_chunk_ids=cited_chunk_ids,
            supporting_evidence_summary=supporting_evidence_summary,
            confidence=confidence,
            confidence_notes=confidence_notes,
            follow_up_questions=self._normalize_string_list(payload.get("follow_up_questions", [])),
            model_name=model_name,
        )

    def _parse_model_output(self, model_output: str) -> dict[str, Any]:
        """Parse the raw model output into the structured response contract.

        Why this function exists:
            Prompt assembly asks the local model to emit JSON so answer assembly
            can validate citations deterministically. This helper centralizes the
            JSON parsing step and makes failures explicit when the local model
            does not respect the contract.

        Parameters:
            model_output: Raw text returned by the local model backend.

        Returns:
            A decoded JSON object representing the model output.

        Edge cases handled:
            Non-JSON output and non-object JSON values raise ``ValueError``
            because the generator must fail closed when the model stops
            following the explicit response contract.
        """
        try:
            payload = json.loads(model_output)
        except json.JSONDecodeError as exc:
            raise ValueError("Local model output must be valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Local model output JSON must be an object.")

        return payload

    def _normalize_cited_chunk_ids(self, value: Any) -> list[str]:
        """Normalize and validate cited chunk ids returned by the model.

        Why this function exists:
            Citation preservation depends on chunk ids matching the retrieved
            evidence exactly. This helper enforces that the model returns a list
            of explicit chunk ids instead of implied or free-form references.

        Parameters:
            value: Candidate ``cited_chunk_ids`` field from the decoded model
                output.

        Returns:
            A list of trimmed chunk-id strings preserving order.

        Edge cases handled:
            Non-list values or blank chunk ids raise explicit errors because the
            generation layer must not guess what the model intended to cite.
        """
        if not isinstance(value, list):
            raise ValueError("cited_chunk_ids must be a list of chunk ids.")

        normalized_chunk_ids: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Every cited chunk id must be a non-empty string.")
            normalized_chunk_ids.append(item.strip())
        return normalized_chunk_ids

    def _resolve_citations(
        self,
        cited_chunk_ids: list[str],
        evidence_by_chunk_id: dict[str, RetrievalResult],
    ) -> list[Citation]:
        """Resolve cited chunk ids back to the structured retrieval citations.

        Why this function exists:
            The generation layer must preserve citations that already exist in
            retrieval results instead of reconstructing them from raw text. This
            helper guarantees that final citations come directly from the
            retrieved evidence and raises if the model cites anything outside
            that closed set.

        Parameters:
            cited_chunk_ids: Chunk ids explicitly returned by the local model.
            evidence_by_chunk_id: Mapping from retrieved chunk ids to retrieval
                results.

        Returns:
            A list of citation objects copied from the retrieved evidence.

        Edge cases handled:
            Unknown chunk ids raise ``ValueError`` immediately so the generator
            cannot silently invent or distort source references.
        """
        citations: list[Citation] = []
        for chunk_id in cited_chunk_ids:
            if chunk_id not in evidence_by_chunk_id:
                raise ValueError(f"Local model cited unknown chunk id: {chunk_id}")
            citations.append(evidence_by_chunk_id[chunk_id].citation)
        return citations

    def _build_supporting_evidence_summary(
        self,
        cited_chunk_ids: list[str],
        evidence_by_chunk_id: dict[str, RetrievalResult],
    ) -> list[str]:
        """Summarize the evidence actually used in the final answer.

        Why this function exists:
            Callers often need a compact explanation of which evidence supported
            the answer without re-reading every full chunk. This helper builds a
            small deterministic summary from the cited retrieval hits only,
            which keeps evidence tracking explicit and avoids implying support
            from uncited chunks.

        Parameters:
            cited_chunk_ids: Chunk ids explicitly used by the model.
            evidence_by_chunk_id: Mapping of retrieved evidence items.

        Returns:
            A list of short evidence summary strings aligned with the cited
            chunk order.

        Edge cases handled:
            If no citations are returned, the summary list remains empty rather
            than fabricating implied supporting evidence from the broader
            retrieval set.
        """
        summaries: list[str] = []
        for chunk_id in cited_chunk_ids:
            result = evidence_by_chunk_id[chunk_id]
            summaries.append(
                f"{result.title} ({result.citation.locator}): {self._truncate_text(result.text)}"
            )
        return summaries

    def _derive_confidence(
        self,
        *,
        cited_chunk_ids: list[str],
        evidence_by_chunk_id: dict[str, RetrievalResult],
        confidence_notes: list[str],
    ) -> float | None:
        """Derive a lightweight confidence estimate from grounded evidence usage.

        Why this function exists:
            The generation layer needs an optional confidence value even when a
            local model backend does not return one directly. This helper uses a
            simple deterministic heuristic based on cited evidence scores and
            confidence notes so the response can surface uncertainty without
            pretending to compute a learned probability.

        Parameters:
            cited_chunk_ids: Chunk ids explicitly cited by the model.
            evidence_by_chunk_id: Retrieved evidence mapping used to recover
                fused retrieval scores.
            confidence_notes: Notes emitted by the model about uncertainty or
                missing evidence.

        Returns:
            A heuristic confidence score in ``0..1`` or ``None`` when there is
            no cited evidence to ground the estimate.

        Edge cases handled:
            Answers without citations return ``None`` because any numeric value
            would imply grounded support that the model did not actually provide.
        """
        if not cited_chunk_ids:
            return None

        average_score = sum(
            evidence_by_chunk_id[chunk_id].score for chunk_id in cited_chunk_ids
        ) / len(cited_chunk_ids)
        confidence_penalty = 0.1 if confidence_notes else 0.0
        return max(0.0, min(1.0, average_score - confidence_penalty))

    def _normalize_string_list(self, value: Any) -> list[str]:
        """Normalize optional list fields returned by the local model.

        Why this function exists:
            The model response contract includes optional lists such as
            confidence notes and follow-up questions. This helper keeps the
            parsing behavior consistent without mixing those generic list checks
            into citation-specific validation logic.

        Parameters:
            value: Candidate list field decoded from the model output.

        Returns:
            A list of trimmed strings.

        Edge cases handled:
            Missing values become an empty list, while malformed non-list values
            raise ``ValueError`` because the model is expected to follow the
            explicit JSON contract.
        """
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Expected a list of strings in local model output.")

        normalized_items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("Expected every list item in local model output to be a string.")
            stripped_item = item.strip()
            if stripped_item:
                normalized_items.append(stripped_item)
        return normalized_items

    def _require_non_empty_string(self, value: Any, *, field_name: str) -> str:
        """Require a non-empty string field from the model output.

        Why this function exists:
            Certain model output fields such as the final answer are mandatory
            for the generation layer. This helper makes those requirements
            explicit and keeps validation errors consistent across fields.

        Parameters:
            value: Candidate field value from the parsed model output.
            field_name: Human-readable field name used in validation errors.

        Returns:
            A stripped non-empty string.

        Edge cases handled:
            Missing, non-string, or blank values raise ``ValueError`` because
            answer assembly cannot construct a valid response without them.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Local model output field '{field_name}' must be a non-empty string.")
        return value.strip()

    def _truncate_text(self, text: str) -> str:
        """Trim evidence text for compact summary display.

        Why this function exists:
            Supporting evidence summaries should stay compact while still
            exposing enough of the cited evidence to be useful in logs, tests,
            and UI surfaces. This helper applies one consistent truncation rule.

        Parameters:
            text: Full retrieved chunk text.

        Returns:
            A compact summary string no longer than the configured citation
            character limit.

        Edge cases handled:
            Short evidence text is returned unchanged, while longer text is
            truncated with an ellipsis to make the incompleteness explicit.
        """
        limit = self._settings.citation_character_limit
        if len(text) <= limit:
            return text
        return f"{text[: max(0, limit - 3)].rstrip()}..."


class GenerationService:
    """Orchestrate prompt assembly, local-model invocation, and answer assembly."""

    def __init__(
        self,
        *,
        prompt_assembly: PromptAssemblyService | None = None,
        local_model_service: LocalModelService | None = None,
        answer_assembly: AnswerAssemblyService | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Store the generation-layer dependencies behind explicit extension points.

        Why this function exists:
            The generation layer should remain modular. Prompt construction,
            local backend invocation, and response assembly are separate
            responsibilities, and wiring them together here keeps the public
            service small without collapsing those concerns into one function.

        Parameters:
            prompt_assembly: Optional prompt assembly service override.
            local_model_service: Optional local model invocation service
                override.
            answer_assembly: Optional answer assembly service override.
            settings: Optional validated settings object shared across
                generation concerns.

        Returns:
            This initializer stores the generation dependencies on the service.

        Edge cases handled:
            Default services are created automatically so the standard local
            generation path remains easy to instantiate in scripts and tests.
        """
        self._settings = settings or get_settings()
        self._prompt_assembly = prompt_assembly or PromptAssemblyService()
        self._local_model_service = local_model_service or LocalModelService(settings=self._settings)
        self._answer_assembly = answer_assembly or AnswerAssemblyService(settings=self._settings)

    def answer(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a grounded answer from retrieved evidence using a local backend.

        Why this function exists:
            This is the public generation entry point. It assembles the prompt,
            calls the configured local model backend, and then validates and
            assembles the final response so citation tracking remains explicit
            from end to end.

        Parameters:
            request: Generation request containing the user query, task type,
                retrieved evidence, and output controls.

        Returns:
            A ``GenerationResponse`` with verified citations and evidence
            summaries.

        Edge cases handled:
            The method does not attempt to repair malformed model citations or
            invalid JSON output because failing closed is safer than inventing
            unsupported grounding metadata.
        """
        prompt = self._prompt_assembly.assemble_prompt(request)
        model_output = self._local_model_service.generate(
            prompt=prompt,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )
        return self._answer_assembly.assemble_answer(
            request=request,
            model_output=model_output,
            model_name=self._local_model_service.model_name(),
        )

    def uses_fallback_backend(self) -> bool:
        """Report whether generation is currently using a deterministic fallback.

        Why this function exists:
            API callers and trace metadata need to distinguish between a real
            local-model answer and a placeholder answer path. Exposing that
            through the service keeps route handlers simple and avoids backend
            introspection there.

        Parameters:
            This helper takes no parameters because it reflects the local model
            service already attached to the generation service.

        Returns:
            ``True`` when the configured local model service is using a fallback
            backend instead of a live provider adapter.

        Edge cases handled:
            Custom backends that do not expose explicit fallback metadata are
            treated as non-fallback providers by the local model service.
        """
        return self._local_model_service.uses_fallback_backend()

    def backend_kind(self) -> str:
        """Return the active generation backend kind for trace metadata.

        Why this function exists:
            Query traces are more useful when they describe which generation
            backend actually ran. This helper forwards that metadata through the
            service boundary.

        Parameters:
            This helper takes no parameters because it reflects the current
            local model service dependency.

        Returns:
            A short backend kind label such as ``ollama`` or ``fallback``.

        Edge cases handled:
            Unknown backend implementations still resolve to a stable fallback
            label via the local model service.
        """
        return self._local_model_service.backend_kind()
