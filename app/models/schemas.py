"""Shared Pydantic schemas for the Local AgentOps scaffold."""

from __future__ import annotations


from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    """Response schema for API health checks."""

    status: str = Field(description="Short health indicator for the running API.")
    service: str = Field(description="Logical service name for the current process.")


class ArchitectureSummary(BaseModel):
    """Response schema describing the high-level system architecture."""

    modules: list[str] = Field(description="Ordered list of major backend modules.")
    retrieval_mode: str = Field(description="Planned retrieval strategy label.")
    llm_mode: str = Field(description="Planned LLM serving mode label.")


class Citation(BaseModel):
    """Concrete citation details used to ground generated answers."""

    chunk_id: str = Field(min_length=1, description="Identifier of the cited chunk.")
    document_id: str = Field(min_length=1, description="Identifier of the parent source document.")
    source_type: Literal["document", "support_ticket"] = Field(
        description="Source family for the cited material."
    )
    title: str = Field(min_length=1, description="Display title shown to a user in the citation.")
    locator: str = Field(
        min_length=1,
        description="Human-readable location such as page, section, or ticket message offset.",
    )
    snippet: str = Field(min_length=1, description="Short excerpt quoted or paraphrased from the source.")
    source_uri: str | None = Field(
        default=None,
        description="Optional original source URL or local URI for drill-down navigation.",
    )


class RawDocument(BaseModel):
    """Raw source artifact accepted by the ingestion layer."""

    document_id: str = Field(min_length=1, description="Stable internal document identifier.")
    external_id: str | None = Field(
        default=None,
        description="Upstream system identifier such as a ticket ID or document GUID.",
    )
    source_type: Literal["document", "support_ticket"] = Field(
        description="Concrete source family used to branch later pipeline behavior."
    )
    connector_name: str = Field(
        min_length=1,
        description="Connector or ingest channel name such as filesystem, zendesk, or csv_import.",
    )
    title: str = Field(min_length=1, description="Best available source title or ticket subject.")
    content: str = Field(min_length=1, description="Raw text payload before parsing and normalization.")
    mime_type: str = Field(
        default="text/plain",
        min_length=1,
        description="Original source mime type used by downstream parsing decisions.",
    )
    source_uri: str | None = Field(
        default=None,
        description="Optional original URL or local path to the source artifact.",
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=16,
        description="Language code for parsing and retrieval decisions.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Source creation timestamp when provided by the upstream system.",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Last-modified timestamp from the upstream system.",
    )
    author_name: str | None = Field(
        default=None,
        description="Optional source author or ticket requester name.",
    )
    customer_id: str | None = Field(
        default=None,
        description="Optional customer or requester identifier for support-ticket sources.",
    )
    ticket_status: Literal["open", "pending", "resolved", "closed"] | None = Field(
        default=None,
        description="Optional ticket workflow status for support-ticket sources.",
    )
    priority: Literal["low", "normal", "high", "urgent"] | None = Field(
        default=None,
        description="Optional priority label for support-ticket sources.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Source tags carried forward for filtering, routing, and evaluation.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Connector-specific fields preserved for later enrichment or debugging.",
    )

    @field_validator("connector_name", "title", "mime_type", "language", "author_name", mode="before")
    @classmethod
    def strip_optional_text(cls, value: object) -> object:
        """Normalize incoming text values by trimming surrounding whitespace.

        Why this function exists:
            Ingestion sources often include padded strings or copied values with
            stray whitespace. Normalizing them early makes IDs and display
            fields more stable across parsers, chunkers, and evaluators.

        Parameters:
            value: Raw field value supplied before Pydantic coercion.

        Returns:
            The same value with outer whitespace removed when the value is a
            string.

        Edge cases handled:
            ``None`` and non-string values are returned unchanged so optional
            fields remain optional and native types still validate normally.
        """
        # Trim incoming strings to avoid carrying noisy whitespace through the
        # pipeline and into chunk IDs or display labels.
        if isinstance(value, str):
            return value.strip()

        # Leave non-string values untouched so standard validation can continue.
        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        """Normalize and deduplicate source tags.

        Why this function exists:
            Tags are likely to be used for filtering and evaluation slices later
            in the project. Normalizing them once prevents subtle mismatches
            caused by case or duplicate values.

        Parameters:
            value: Tag list supplied by ingestion connectors.

        Returns:
            A normalized list of unique lowercase tags preserving first-seen
            order.

        Edge cases handled:
            Empty strings are removed; repeated tags collapse into one entry; an
            empty input list remains valid.
        """
        # Use an ordered dictionary-like pattern via ``set`` membership checks
        # so duplicates are removed while first-seen order stays intact.
        normalized_tags: list[str] = []
        seen_tags: set[str] = set()
        for tag in value:
            normalized_tag = tag.strip().lower()
            if not normalized_tag or normalized_tag in seen_tags:
                continue
            normalized_tags.append(normalized_tag)
            seen_tags.add(normalized_tag)

        # Return the cleaned tag list for downstream filtering and analytics.
        return normalized_tags

    @model_validator(mode="after")
    def validate_source_specific_fields(self) -> "RawDocument":
        """Enforce concrete rules for document and support-ticket sources.

        Why this function exists:
            The model is intentionally concrete so later ingestion and parsing
            stages can rely on support-ticket-only fields when present without
            making every source shape completely bespoke.

        Parameters:
            This validator works against the fully parsed model instance and
            therefore takes no extra parameters.

        Returns:
            The same ``RawDocument`` instance after cross-field validation.

        Edge cases handled:
            Support tickets may omit customer metadata for some imports, but if
            ticket-only fields are supplied for general documents the model still
            accepts them to support mixed-source backfills and connector quirks.
        """
        # Ensure timestamps move forward logically when both are present.
        if self.created_at and self.updated_at and self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")

        # Return the validated model instance for normal use.
        return self


class ParsedSection(BaseModel):
    """Structured subsection extracted from a raw document."""

    heading: str = Field(min_length=1, description="Section heading or synthetic label.")
    text: str = Field(min_length=1, description="Normalized plain text for the section.")
    order: int = Field(ge=0, description="Zero-based order of the section in the parsed document.")
    start_offset: int = Field(ge=0, description="Character offset where the section starts in plain text.")
    end_offset: int = Field(ge=0, description="Character offset where the section ends in plain text.")

    @model_validator(mode="after")
    def validate_offsets(self) -> "ParsedSection":
        """Validate section offsets produced by parsing.

        Why this function exists:
            Retrieval and citation assembly often depend on stable character
            offsets. Catching broken offsets early prevents chunking and answer
            grounding bugs later in the pipeline.

        Parameters:
            This validator operates on the parsed section instance itself.

        Returns:
            The same ``ParsedSection`` instance after verification.

        Edge cases handled:
            Zero-length sections are rejected because they produce unusable
            chunks and confusing citations.
        """
        # Ensure the section end is strictly after the start so the range is
        # meaningful for later citation construction.
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")

        # Return the validated section model for further processing.
        return self


class ParsedDocument(BaseModel):
    """Normalized representation produced by the parsing stage."""

    document_id: str = Field(min_length=1, description="Stable internal identifier inherited from raw input.")
    source_type: Literal["document", "support_ticket"] = Field(
        description="Source family carried forward for later routing and metrics."
    )
    title: str = Field(min_length=1, description="Normalized title or ticket subject.")
    plain_text: str = Field(min_length=1, description="Normalized full plain-text representation.")
    summary: str | None = Field(
        default=None,
        description="Optional parser-produced summary used for previews or debugging.",
    )
    language: str = Field(min_length=2, max_length=16, description="Detected or inherited language code.")
    sections: list[ParsedSection] = Field(
        default_factory=list,
        description="Ordered parsed sections derived from the source content.",
    )
    source_uri: str | None = Field(
        default=None,
        description="Original URL or path carried forward for traceability.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured parser metadata such as page counts or extracted headers.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Original source creation time when available.",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Original source update time when available.",
    )

    @model_validator(mode="after")
    def validate_sections_against_text(self) -> "ParsedDocument":
        """Ensure parsed sections align with the normalized plain text.

        Why this function exists:
            The chunking layer will rely on section offsets to preserve document
            structure. Validating them here makes later chunking behavior easier
            to reason about and debug.

        Parameters:
            This validator uses the parsed document instance directly and takes
            no extra parameters.

        Returns:
            The same ``ParsedDocument`` instance once section ranges are known
            to fit inside ``plain_text``.

        Edge cases handled:
            Documents without sections remain valid because some parsers may
            only provide a flat text body during early implementation.
        """
        # Check each section range against the overall text length so chunking
        # and citation generation do not index past the end of the document.
        text_length = len(self.plain_text)
        for section in self.sections:
            if section.end_offset > text_length:
                raise ValueError("section end_offset cannot exceed plain_text length")

        # Return the validated parsed document to the caller.
        return self


class DocumentChunk(BaseModel):
    """Retrieval-ready chunk derived from a parsed document."""

    chunk_id: str = Field(min_length=1, description="Stable chunk identifier used across indexes.")
    document_id: str = Field(min_length=1, description="Parent document identifier.")
    source_type: Literal["document", "support_ticket"] = Field(
        description="Parent source family used for filtering and downstream formatting."
    )
    text: str = Field(min_length=1, description="Chunk text stored in lexical and vector indexes.")
    token_count_estimate: int = Field(
        ge=0,
        description="Approximate token count for retrieval packing and LLM context budgeting.",
    )
    char_start: int = Field(ge=0, description="Inclusive character offset within the parsed plain text.")
    char_end: int = Field(ge=0, description="Exclusive character offset within the parsed plain text.")
    chunk_index: int = Field(ge=0, description="Zero-based chunk position within the parent document.")
    section_heading: str | None = Field(
        default=None,
        description="Nearest parsed section heading used to improve citations and prompts.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk-specific metadata such as page number, speaker, or parser hints.",
    )

    @model_validator(mode="after")
    def validate_chunk_offsets(self) -> "DocumentChunk":
        """Verify that chunk offsets form a usable span.

        Why this function exists:
            Retrieval results and citations refer back to character spans in the
            parsed document. Invalid spans here would create broken evidence
            links and confusing answer citations.

        Parameters:
            This validator works on the fully parsed chunk instance itself.

        Returns:
            The same ``DocumentChunk`` instance after offset verification.

        Edge cases handled:
            Empty spans are rejected even if the chunk text itself is present,
            because later highlighting logic depends on a valid range.
        """
        # Ensure the character span is positive so the chunk points at a real
        # slice of the parsed document.
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")

        # Return the validated chunk model for indexing and retrieval.
        return self


class RetrievalRequest(BaseModel):
    """Concrete retrieval request used by the hybrid search layer."""

    query: str = Field(min_length=3, description="User question or search statement.")
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum number of ranked results to return.")
    source_types: list[Literal["document", "support_ticket"]] = Field(
        default_factory=list,
        description="Optional source-type filters for retrieval narrowing.",
    )
    tag_filters: list[str] = Field(
        default_factory=list,
        description="Optional normalized tag filters applied during candidate selection.",
    )
    include_archived: bool = Field(
        default=False,
        description="Whether archived or closed knowledge should be considered when available.",
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional relevance threshold applied after ranking normalization.",
    )

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: object) -> object:
        """Trim query strings before validation.

        Why this function exists:
            Retrieval quality suffers when blank or whitespace-padded queries
            make it into later ranking steps. Cleaning them here makes the
            request contract reliable.

        Parameters:
            value: Raw query value supplied to the model.

        Returns:
            The stripped string when the input is textual, otherwise the
            original value for normal validation.

        Edge cases handled:
            Non-string values are passed through so Pydantic can emit a clear
            type error instead of failing unexpectedly in custom code.
        """
        # Remove leading and trailing whitespace from textual queries.
        if isinstance(value, str):
            return value.strip()

        # Preserve non-string values for Pydantic's standard error handling.
        return value


class RetrievalResult(BaseModel):
    """Ranked retrieval hit returned from hybrid search."""

    chunk_id: str = Field(min_length=1, description="Identifier of the matched chunk.")
    document_id: str = Field(min_length=1, description="Identifier of the parent document.")
    source_type: Literal["document", "support_ticket"] = Field(
        description="Source family for result filtering and display logic."
    )
    score: float = Field(ge=0.0, le=1.0, description="Normalized fused relevance score.")
    rank: int = Field(ge=1, description="One-based rank position after result sorting.")
    text: str = Field(min_length=1, description="Chunk text returned for answer grounding and debug views.")
    title: str = Field(min_length=1, description="Parent document title or ticket subject.")
    section_heading: str | None = Field(
        default=None,
        description="Nearest section label for UI display and citation formatting.",
    )
    source_uri: str | None = Field(
        default=None,
        description="Original source location for drill-down or UI linking.",
    )
    document_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document- and chunk-level metadata surfaced directly to retrieval callers.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Retriever-specific metrics such as bm25 or vector sub-scores.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured provenance describing how the retriever found and merged this hit.",
    )
    citation: Citation = Field(description="Concrete citation payload derived from the retrieval hit.")


class OrchestrationRequest(BaseModel):
    """Typed user request accepted by the deterministic query router."""

    user_request: str = Field(min_length=3, description="Raw user request that should be routed.")

    @field_validator("user_request", mode="before")
    @classmethod
    def strip_user_request(cls, value: object) -> object:
        """Trim incoming request text before routing.

        Why this function exists:
            Routing rules should operate on normalized user input so accidental
            whitespace does not create confusing classification behavior or
            brittle tests.

        Parameters:
            value: Raw request value supplied by the caller.

        Returns:
            The stripped string when the input is textual, otherwise the
            original value for standard validation.

        Edge cases handled:
            Non-string inputs are preserved so Pydantic can emit the usual type
            validation error instead of failing inside router logic.
        """
        if isinstance(value, str):
            return value.strip()

        return value


class TaskClassification(BaseModel):
    """Transparent classification decision emitted by the query router."""

    task_type: Literal[
        "document_qa",
        "policy_lookup",
        "ticket_similarity",
        "summarization",
        "field_extraction",
        "reply_drafting",
    ] = Field(description="Deterministic task type selected by the query router.")
    matched_rules: list[str] = Field(
        default_factory=list,
        description="Human-readable routing rules that contributed to the selected task type.",
    )
    alternative_task_types: list[str] = Field(
        default_factory=list,
        description="Other candidate task types considered but not selected as the primary route.",
    )
    routing_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Heuristic confidence score derived from matched routing rules.",
    )


class OrchestrationStep(BaseModel):
    """One deterministic pipeline step in an orchestration plan."""

    step_id: str = Field(min_length=1, description="Stable identifier for the step within the plan.")
    component: str = Field(min_length=1, description="Pipeline component that should be called.")
    action: str = Field(min_length=1, description="Short action description for the component call.")
    purpose: str = Field(min_length=1, description="Why this step exists in the selected task flow.")
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured inputs that the orchestrator would pass into the component.",
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="Named outputs expected from the step for downstream consumption.",
    )


class OrchestrationPlan(BaseModel):
    """Structured plan describing the deterministic orchestration path for a request."""

    user_request: str = Field(min_length=1, description="Original user request being orchestrated.")
    classification: TaskClassification = Field(
        description="Transparent task-classification decision used to drive orchestration."
    )
    execution_mode: Literal["deterministic_plan"] = Field(
        default="deterministic_plan",
        description="Planner mode indicating that the output is a fixed orchestration plan, not agent behavior.",
    )
    components_to_call: list[str] = Field(
        default_factory=list,
        description="Ordered list of component names that the orchestrator selected for this task.",
    )
    steps: list[OrchestrationStep] = Field(
        default_factory=list,
        description="Ordered execution steps describing the chosen orchestration path.",
    )
    completion_artifact: str = Field(
        min_length=1,
        description="Short description of the final output that the pipeline should produce for the task.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Additional deterministic orchestration notes, such as skipped components or fallback behavior.",
    )


class IngestSourcesRequest(BaseModel):
    """HTTP request contract for ingesting source directories or inline documents."""

    input_directories: list[str] = Field(
        default_factory=list,
        description="Filesystem directories that should be scanned for ingestible sources.",
    )
    documents: list[RawDocument] = Field(
        default_factory=list,
        description="Inline raw documents that should be processed without filesystem discovery.",
    )
    recursive: bool = Field(
        default=True,
        description="Whether filesystem discovery should recurse into nested directories.",
    )


class IngestSourcesResponse(BaseModel):
    """HTTP response describing the result of an ingestion run."""

    ingested_document_ids: list[str] = Field(
        default_factory=list,
        description="Document ids registered and processed during the ingestion request.",
    )
    parsed_document_ids: list[str] = Field(
        default_factory=list,
        description="Parsed document ids persisted for downstream chunking and indexing.",
    )
    chunk_count: int = Field(ge=0, description="Number of chunks persisted during the request.")
    issues: list[dict[str, str]] = Field(
        default_factory=list,
        description="Flattened ingest issues recorded during filesystem discovery or loading.",
    )
    trace_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Trace metadata describing how the API request mapped to ingestion, parsing, and chunking.",
    )


class IndexRefreshRequest(BaseModel):
    """HTTP request contract for building or refreshing a local index collection."""

    collection_name: str = Field(
        default="default",
        min_length=1,
        description="Logical collection name that should receive the refreshed index manifests.",
    )
    include_lexical: bool = Field(
        default=True,
        description="Whether lexical inputs should be refreshed alongside vector inputs.",
    )


class IndexRefreshResponse(BaseModel):
    """HTTP response describing the result of an index refresh operation."""

    collection_name: str = Field(min_length=1, description="Logical collection that was refreshed.")
    chunk_count: int = Field(ge=0, description="Number of chunks included in the refreshed collection.")
    vector_manifest_path: str = Field(min_length=1, description="Filesystem path of the vector manifest.")
    lexical_manifest_path: str | None = Field(
        default=None,
        description="Filesystem path of the lexical manifest when lexical indexing is enabled.",
    )
    metadata_manifest_path: str = Field(
        min_length=1,
        description="Filesystem path of the collection metadata manifest.",
    )
    trace_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Trace metadata describing how the refresh request mapped to indexing operations.",
    )


class RetrievalExecutionRequest(BaseModel):
    """HTTP request contract for retrieval-only API calls."""

    query: str = Field(min_length=3, description="User query that should be run through retrieval.")
    collection_name: str = Field(
        default="default",
        min_length=1,
        description="Logical collection name that should be queried.",
    )
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum number of retrieval results to return.")
    source_types: list[Literal["document", "support_ticket"]] = Field(
        default_factory=list,
        description="Optional source-type filters applied before retrieval ranking.",
    )
    tag_filters: list[str] = Field(
        default_factory=list,
        description="Optional tag filters applied before retrieval ranking.",
    )
    include_archived: bool = Field(
        default=False,
        description="Whether archived sources should remain eligible during retrieval.",
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional fused-score threshold applied after ranking.",
    )
    enable_hybrid: bool = Field(
        default=True,
        description="Whether lexical retrieval should be included when the collection has lexical inputs.",
    )


class RetrievalExecutionResponse(BaseModel):
    """HTTP response describing a retrieval-only request."""

    query: str = Field(min_length=1, description="Original query executed by the endpoint.")
    collection_name: str = Field(min_length=1, description="Logical collection that was queried.")
    results: list[RetrievalResult] = Field(
        default_factory=list,
        description="Ranked retrieval results returned by the retrieval layer.",
    )
    trace_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Trace metadata describing the retrieval execution path and settings.",
    )


class QueryExecutionRequest(BaseModel):
    """HTTP request contract for orchestrated query execution."""

    query: str = Field(min_length=3, description="User query that should run through orchestration.")
    collection_name: str = Field(
        default="default",
        min_length=1,
        description="Logical collection name used by retrieval during query execution.",
    )
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum number of retrieval results to use.")
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional retrieval threshold applied before answer generation.",
    )
    include_archived: bool = Field(
        default=False,
        description="Whether archived sources should remain eligible during retrieval.",
    )
    output_instructions: list[str] = Field(
        default_factory=list,
        description="Optional deterministic answer-formatting instructions passed to generation.",
    )


class QueryExecutionResponse(BaseModel):
    """HTTP response describing one end-to-end orchestrated query execution."""

    query: str = Field(min_length=1, description="Original query submitted to the endpoint.")
    task_type: str = Field(min_length=1, description="Task type selected by the deterministic router.")
    orchestration_plan: OrchestrationPlan = Field(
        description="Structured orchestration plan selected for the query."
    )
    retrieval_results: list[RetrievalResult] = Field(
        default_factory=list,
        description="Ranked evidence returned by the retrieval layer.",
    )
    answer: GenerationResponse | None = Field(
        default=None,
        description="Final grounded answer when the selected task path includes generation.",
    )
    trace_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Trace metadata describing orchestration, retrieval, and generation behavior.",
    )


class GenerationRequest(BaseModel):
    """Input contract for grounded answer generation."""

    query: str = Field(min_length=3, description="User question that requires a grounded answer.")
    task_type: Literal[
        "document_qa",
        "policy_lookup",
        "ticket_similarity",
        "summarization",
        "field_extraction",
        "reply_drafting",
    ] = Field(
        default="document_qa",
        description="Deterministic task type used to shape prompt instructions and answer formatting.",
    )
    retrieved_context: list[RetrievalResult] = Field(
        min_length=1,
        description="Ranked retrieval hits to inject into the generation prompt.",
    )
    output_instructions: list[str] = Field(
        default_factory=list,
        description="Additional deterministic output instructions appended to the assembled prompt.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional override for the default system prompt used by the local LLM.",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature for generation.")
    max_output_tokens: int = Field(
        default=512,
        ge=32,
        le=4096,
        description="Maximum answer length budget reserved for the local LLM.",
    )
    include_follow_up_questions: bool = Field(
        default=False,
        description="Whether the model should propose next questions after the answer.",
    )

    @field_validator("query", "system_prompt", mode="before")
    @classmethod
    def strip_request_text(cls, value: object) -> object:
        """Trim generation text fields before validation.

        Why this function exists:
            Prompt construction is sensitive to accidental whitespace-only
            strings. Cleaning those values here makes prompt assembly more
            predictable later on.

        Parameters:
            value: Raw field value supplied to the model.

        Returns:
            The stripped string when the value is textual, otherwise the
            original value.

        Edge cases handled:
            ``None`` remains ``None`` for optional fields, and non-string values
            continue through standard validation untouched.
        """
        # Normalize surrounding whitespace for text-based prompt fields.
        if isinstance(value, str):
            return value.strip()

        # Return non-string values unchanged so Pydantic can validate them.
        return value

    @field_validator("output_instructions")
    @classmethod
    def normalize_output_instructions(cls, value: list[str]) -> list[str]:
        """Normalize optional output instructions passed into prompt assembly.

        Why this function exists:
            Prompt construction should operate on clean instruction strings
            rather than carrying forward empty list entries or stray
            whitespace. Normalizing those instructions here keeps the prompt
            assembly path deterministic.

        Parameters:
            value: Output instruction strings supplied by the caller.

        Returns:
            A cleaned list of non-empty instruction strings preserving order.

        Edge cases handled:
            Empty or whitespace-only instructions are removed so the prompt
            builder does not emit blank bullets that might confuse a local
            model.
        """
        return [instruction.strip() for instruction in value if instruction.strip()]


class GenerationResponse(BaseModel):
    """Grounded answer payload returned by the generation layer."""

    answer: str = Field(min_length=1, description="Final natural-language answer shown to the user.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations supporting the statements made in the answer.",
    )
    used_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk identifiers actually incorporated into the answer prompt or output.",
    )
    supporting_evidence_summary: list[str] = Field(
        default_factory=list,
        description="Compact summaries of the evidence chunks actually cited or used in the answer.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional model- or heuristic-derived confidence score.",
    )
    confidence_notes: list[str] = Field(
        default_factory=list,
        description="Optional notes explaining uncertainty, ambiguity, or missing evidence in the answer.",
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Optional next questions suggested by the generation layer.",
    )
    model_name: str = Field(min_length=1, description="Local LLM model name used to generate the answer.")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp describing when the response was produced.",
    )

    @field_validator("follow_up_questions")
    @classmethod
    def normalize_follow_up_questions(cls, value: list[str]) -> list[str]:
        """Clean optional follow-up questions emitted by the model.

        Why this function exists:
            Generation outputs can include duplicates or blank suggestions.
            Normalizing them here keeps the API contract clean for UI and
            evaluation consumers.

        Parameters:
            value: Follow-up question list emitted by the generator.

        Returns:
            A list of non-empty trimmed questions preserving order.

        Edge cases handled:
            Blank strings are removed; duplicated questions are preserved for
            now because repetition itself may be a useful evaluation signal.
        """
        # Remove empty or whitespace-only question strings while preserving
        # original order for downstream display or analysis.
        normalized_questions = [question.strip() for question in value if question.strip()]

        # Return the cleaned list to the caller.
        return normalized_questions

    @field_validator("supporting_evidence_summary", "confidence_notes")
    @classmethod
    def normalize_text_lists(cls, value: list[str]) -> list[str]:
        """Normalize generated text lists kept on the response payload.

        Why this function exists:
            The generation layer returns several optional lists that may contain
            blank strings depending on model formatting. Normalizing them in the
            schema keeps the public contract clean for API callers and tests.

        Parameters:
            value: List of generated text snippets such as evidence summaries or
                confidence notes.

        Returns:
            A list of non-empty trimmed strings.

        Edge cases handled:
            Blank entries are removed while preserving order so the response can
            still reflect the model's intended sequence without exposing empty
            artifacts.
        """
        return [item.strip() for item in value if item.strip()]


class EvaluationRecord(BaseModel):
    """Concrete evaluation record for retrieval and generation experiments."""

    evaluation_id: str = Field(min_length=1, description="Stable identifier for the evaluation run or example.")
    dataset_name: str = Field(min_length=1, description="Benchmark dataset or fixture collection name.")
    query: str = Field(min_length=1, description="Prompt or question evaluated in this record.")
    expected_answer: str | None = Field(
        default=None,
        description="Optional gold answer text for answer-quality comparisons.",
    )
    expected_document_ids: list[str] = Field(
        default_factory=list,
        description="Known relevant documents for retrieval quality scoring.",
    )
    retrieved_document_ids: list[str] = Field(
        default_factory=list,
        description="Documents actually returned by the retriever for this query.",
    )
    generated_answer: str | None = Field(
        default=None,
        description="Answer returned by the generation pipeline during evaluation.",
    )
    retrieval_precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Precision-style retrieval score for the evaluated query.",
    )
    retrieval_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Recall-style retrieval score for the evaluated query.",
    )
    groundedness_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How well the generated answer stays supported by retrieved evidence.",
    )
    answer_correctness_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How well the generated answer matches the expected answer or rubric.",
    )
    notes: str | None = Field(
        default=None,
        description="Free-form evaluator comments or failure analysis notes.",
    )
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp indicating when the evaluation record was captured.",
    )


class ValidationOutcome(BaseModel):
    """Structured result from one deterministic validation rule."""

    rule_name: str = Field(min_length=1, description="Stable validation rule identifier.")
    passed: bool = Field(description="Whether the validation rule passed.")
    severity: Literal["info", "warning", "error"] = Field(
        description="Severity level describing the impact of a failed validation rule."
    )
    details: str = Field(min_length=1, description="Human-readable explanation of the validation result.")


class EvaluationTestCase(BaseModel):
    """One deterministic evaluation case executed by the benchmark harness."""

    evaluation_id: str = Field(min_length=1, description="Stable identifier for the benchmark case.")
    query: str = Field(min_length=1, description="User query or task input evaluated by the harness.")
    task_type: Literal[
        "document_qa",
        "policy_lookup",
        "ticket_similarity",
        "summarization",
        "field_extraction",
        "reply_drafting",
    ] = Field(
        default="document_qa",
        description="Deterministic task type used to shape prompt assembly for the case.",
    )
    retrieved_context: list[RetrievalResult] = Field(
        default_factory=list,
        description="Retrieved evidence provided to the generation layer for this case.",
    )
    output_instructions: list[str] = Field(
        default_factory=list,
        description="Optional deterministic output instructions for the generation request.",
    )
    expected_answer: str | None = Field(
        default=None,
        description="Optional expected answer text carried for later manual review or future scoring.",
    )


class EvaluationRunRecord(BaseModel):
    """Auditable record emitted for one harness execution case."""

    evaluation_id: str = Field(min_length=1, description="Stable identifier for the benchmark case.")
    dataset_name: str = Field(min_length=1, description="Dataset or benchmark collection name.")
    query: str = Field(min_length=1, description="Query executed during the evaluation run.")
    task_type: str = Field(min_length=1, description="Task type used for prompt assembly and planning.")
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk ids present in the retrieved context passed to generation.",
    )
    selected_evidence_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk ids explicitly selected by the generator in the final answer.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Final answer text returned by the generation layer when generation succeeded.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Structured citations preserved in the final answer payload.",
    )
    supporting_evidence_summary: list[str] = Field(
        default_factory=list,
        description="Compact summaries of the evidence cited in the answer.",
    )
    latency_ms: float = Field(
        ge=0.0,
        description="Wall-clock latency for the generation-plus-validation path in milliseconds.",
    )
    validation_outcomes: list[ValidationOutcome] = Field(
        default_factory=list,
        description="Deterministic validation results recorded for the run.",
    )
    passed: bool = Field(description="Whether all error-severity validation rules passed.")
    notes: list[str] = Field(
        default_factory=list,
        description="Additional benchmark notes such as generation failures or missing evidence.",
    )
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp indicating when the evaluation run record was captured.",
    )


class EvaluationExecutionRequest(BaseModel):
    """HTTP request contract for running a deterministic evaluation suite."""

    dataset_name: str = Field(min_length=1, description="Logical dataset name used for saved evaluation results.")
    test_cases: list[EvaluationTestCase] = Field(
        default_factory=list,
        description="Ordered deterministic evaluation cases to execute.",
    )
    save_results: bool = Field(
        default=True,
        description="Whether evaluation results should be persisted under the managed eval directory.",
    )


class EvaluationExecutionResponse(BaseModel):
    """HTTP response describing one evaluation harness run."""

    dataset_name: str = Field(min_length=1, description="Logical dataset name that was evaluated.")
    records: list[EvaluationRunRecord] = Field(
        default_factory=list,
        description="Per-case evaluation records emitted by the harness.",
    )
    saved_path: str | None = Field(
        default=None,
        description="Filesystem path of the saved evaluation report when persistence is enabled.",
    )
    trace_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Trace metadata describing benchmark execution and persistence behavior.",
    )
