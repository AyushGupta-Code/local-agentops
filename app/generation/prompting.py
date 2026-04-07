"""Prompt assembly helpers for grounded local generation."""

from app.models.schemas import GenerationRequest, RetrievalResult


class PromptAssemblyService:
    """Build deterministic prompts from the user request and retrieved evidence."""

    def assemble_prompt(self, request: GenerationRequest) -> str:
        """Assemble the final prompt sent to the local model backend.

        Why this function exists:
            The generation layer should not inline prompt formatting inside the
            model-calling service because prompt construction is itself a
            distinct, testable responsibility. This method combines the user
            question, task type, retrieved evidence, citation metadata, and
            output instructions into one deterministic prompt string.

        Parameters:
            request: Generation request containing the question, task type, and
                evidence selected by retrieval.

        Returns:
            A fully assembled prompt string ready for a local model backend.

        Edge cases handled:
            The method always includes explicit evidence and citation sections,
            even for small evidence sets, so the model sees a consistent prompt
            format and cannot claim that citations were omitted implicitly.
        """
        evidence_block = self._format_evidence_block(request.retrieved_context)
        instruction_block = self._format_output_instructions(request)
        system_preamble = request.system_prompt or self._default_system_prompt(request.task_type)

        return "\n\n".join(
            [
                system_preamble,
                f"Task type: {request.task_type}",
                f"User question:\n{request.query}",
                "Selected evidence with citation metadata:\n" + evidence_block,
                "Output instructions:\n" + instruction_block,
                self._response_contract(),
            ]
        )

    def _default_system_prompt(self, task_type: str) -> str:
        """Provide the default system instructions for grounded local generation.

        Why this function exists:
            The local model needs a consistent baseline instruction set even
            when the caller does not provide a custom system prompt. The
            baseline prompt makes the grounding and citation requirements
            explicit before any task-specific output instructions are added.

        Parameters:
            task_type: Deterministic task type selected by the orchestrator.

        Returns:
            A system-style instruction string tailored to grounded answer
            generation.

        Edge cases handled:
            The prompt names the task type directly so one local model backend
            can be reused across multiple deterministic task paths without
            hidden routing state.
        """
        return (
            "You are a local grounded generation component.\n"
            f"You are handling task type '{task_type}'.\n"
            "Use only the provided evidence.\n"
            "Never invent citations, document titles, or chunk ids.\n"
            "If the evidence is insufficient, say so explicitly in the answer or confidence notes.\n"
            "Return valid JSON matching the requested response contract."
        )

    def _format_evidence_block(self, retrieved_context: list[RetrievalResult]) -> str:
        """Format retrieved evidence into an explicit citation-preserving prompt section.

        Why this function exists:
            Evidence formatting determines whether the local model can reliably
            ground its answer and preserve source tracking. This method renders
            each retrieval hit with chunk identifiers, document metadata,
            citation locator, and text so the model can reference evidence
            without needing to infer hidden provenance.

        Parameters:
            retrieved_context: Ranked retrieval hits selected upstream.

        Returns:
            A multi-line evidence block ready for inclusion in the final prompt.

        Edge cases handled:
            Every evidence item includes both machine-friendly identifiers and
            human-readable citation fields so downstream answer assembly can map
            cited chunk ids back to structured citation objects without relying
            on brittle text matching.
        """
        return "\n\n".join(
            self._format_single_evidence_item(result)
            for result in retrieved_context
        )

    def _format_single_evidence_item(self, result: RetrievalResult) -> str:
        """Render one retrieval result into a prompt-safe evidence entry.

        Why this function exists:
            Local models respond more predictably when each evidence item is
            rendered in a stable structure. This helper makes the evidence
            representation uniform across answer generation tests and scripts.

        Parameters:
            result: One retrieval result selected for prompt assembly.

        Returns:
            A formatted multi-line string describing the evidence item.

        Edge cases handled:
            Optional metadata such as section heading or source URI is included
            explicitly, with missing values rendered as ``n/a``, so the model
            does not have to guess whether information was omitted or absent.
        """
        return (
            f"Evidence rank: {result.rank}\n"
            f"Chunk id: {result.chunk_id}\n"
            f"Document id: {result.document_id}\n"
            f"Title: {result.title}\n"
            f"Source type: {result.source_type}\n"
            f"Score: {result.score:.3f}\n"
            f"Section: {result.section_heading or 'n/a'}\n"
            f"Locator: {result.citation.locator}\n"
            f"Source URI: {result.source_uri or 'n/a'}\n"
            f"Citation snippet: {result.citation.snippet}\n"
            f"Evidence text: {result.text}"
        )

    def _format_output_instructions(self, request: GenerationRequest) -> str:
        """Build the deterministic output instructions appended to the prompt.

        Why this function exists:
            The prompt needs stable output instructions so the local model knows
            exactly how to format its response for answer assembly. Keeping that
            logic in one helper makes task-specific instructions easy to extend
            without changing evidence formatting or backend integration.

        Parameters:
            request: Generation request whose task type and custom instructions
                should shape the output section.

        Returns:
            A newline-delimited instruction block.

        Edge cases handled:
            Callers can supply additional instructions, but the core grounding
            and citation rules are always prepended so task customization never
            weakens source tracking requirements.
        """
        instructions = [
            self._task_specific_instruction(request.task_type),
            "Cite only chunk ids that appear in the provided evidence.",
            "If no evidence supports a claim, say that the evidence is insufficient.",
            "Keep citations separate from the prose answer by returning cited chunk ids in JSON.",
        ]
        instructions.extend(request.output_instructions)
        return "\n".join(f"- {instruction}" for instruction in instructions)

    def _task_specific_instruction(self, task_type: str) -> str:
        """Return the primary output instruction for the selected task type.

        Why this function exists:
            Different deterministic orchestration tasks expect different output
            shapes from the same local model backend. This helper injects one
            high-signal instruction describing how the answer should be framed
            for the selected task type.

        Parameters:
            task_type: Deterministic task type selected earlier in the pipeline.

        Returns:
            A short instruction string for the local model.

        Edge cases handled:
            Unknown task types fall back to a generic grounded-answer
            instruction so prompt assembly remains robust during incremental
            extension work.
        """
        instructions = {
            "document_qa": "Answer the user's question directly using grounded evidence.",
            "policy_lookup": "Provide a precise policy answer and call out the governing evidence.",
            "ticket_similarity": "Summarize the most relevant similar tickets rather than drafting a new answer.",
            "summarization": "Condense the evidence into a concise summary with key takeaways.",
            "field_extraction": "Return the requested fields based only on the evidence.",
            "reply_drafting": "Draft a user-facing reply grounded in the provided evidence.",
        }
        return instructions.get(task_type, "Provide a grounded answer using the provided evidence.")

    def _response_contract(self) -> str:
        """Describe the exact JSON shape expected back from the local model.

        Why this function exists:
            Answer assembly should work from a deterministic model contract
            rather than scraping prose heuristically. This helper makes the JSON
            schema explicit inside the prompt so mocked tests and real local
            backends can target the same output format.

        Parameters:
            This helper takes no parameters because the response contract is
            fixed for the current generation layer.

        Returns:
            A string describing the required JSON response shape.

        Edge cases handled:
            The contract always requires explicit ``cited_chunk_ids`` so the
            model cannot silently imply citations in free-form prose that the
            answer assembler would be unable to verify.
        """
        return (
            "Respond with JSON using these fields only:\n"
            "{\n"
            '  "answer": string,\n'
            '  "cited_chunk_ids": [string],\n'
            '  "confidence_notes": [string],\n'
            '  "follow_up_questions": [string]\n'
            "}"
        )
