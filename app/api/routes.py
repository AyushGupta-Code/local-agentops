"""API route definitions for the Local AgentOps service."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from app.chunking.service import ChunkingService
from app.core.config import get_settings
from app.evaluation.service import EvaluationService
from app.generation.service import GenerationService
from app.indexing.service import IndexingService
from app.ingestion.service import IngestionService
from app.models.schemas import (
    ArchitectureSummary,
    EvaluationExecutionRequest,
    EvaluationExecutionResponse,
    GenerationRequest,
    HealthResponse,
    IndexRefreshRequest,
    IndexRefreshResponse,
    IngestSourcesRequest,
    IngestSourcesResponse,
    OrchestrationRequest,
    QueryExecutionRequest,
    QueryExecutionResponse,
    RetrievalExecutionRequest,
    RetrievalExecutionResponse,
    RetrievalRequest,
)
from app.orchestration.service import OrchestrationService
from app.parsing.service import ParsingService
from app.retrieval.service import RetrievalService
from app.storage.service import LocalFileStorageService

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return a lightweight health response for local orchestration.

    Why this function exists:
        Operators and tests need a minimal endpoint that proves the FastAPI app
        imported successfully and can answer requests.

    Response structure:
        The endpoint returns a small structured payload with a status string and
        logical service name.

    How the API maps to pipeline modules:
        This endpoint does not execute pipeline modules directly; it only proves
        that the API process itself is available.
    """
    return HealthResponse(status="ok", service="local-agentops-api")


@router.get("/architecture", response_model=ArchitectureSummary)
def get_architecture_summary() -> ArchitectureSummary:
    """Expose a brief summary of the current backend architecture.

    Why this function exists:
        The project is intentionally modular, so a self-describing endpoint is
        useful for developers and automated contributors who need to confirm the
        active pipeline boundaries.

    Response structure:
        The response lists the major modules plus high-level retrieval and LLM
        modes.

    How the API maps to pipeline modules:
        The endpoint describes the modules rather than running them, which keeps
        system introspection separate from operational workflows.
    """
    return ArchitectureSummary(
        modules=[
            "ingestion",
            "parsing",
            "chunking",
            "indexing",
            "retrieval",
            "orchestration",
            "generation",
            "evaluation",
            "storage",
        ],
        retrieval_mode="hybrid",
        llm_mode="local-first",
    )


@router.post("/ingest/sources", response_model=IngestSourcesResponse)
def ingest_sources(payload: IngestSourcesRequest) -> IngestSourcesResponse:
    """Run ingestion, parsing, chunking, and local persistence for supplied sources.

    Why this function exists:
        The ingestion endpoint is the API entry point for getting source
        material into the local-first pipeline. It accepts either filesystem
        directories or inline raw documents so both local demos and automated
        tests can drive the pipeline without extra setup.

    Response structure:
        The response returns ingested and parsed document ids, total chunk count,
        flattened ingest issues, and trace metadata describing how the request
        mapped onto ingestion, parsing, chunking, and storage.

    How the API maps to pipeline modules:
        The endpoint first calls ``IngestionService`` for filesystem discovery,
        then ``ParsingService`` to normalize raw documents, ``ChunkingService``
        to create retrieval-ready chunks, and ``LocalFileStorageService`` to
        persist parsed documents and chunks for later indexing.
    """
    if not payload.input_directories and not payload.documents:
        raise HTTPException(status_code=400, detail="Provide input_directories or documents.")

    settings = get_settings()
    ingestion_service = IngestionService()
    parsing_service = ParsingService()
    chunking_service = ChunkingService()
    storage = LocalFileStorageService(settings)

    raw_documents = []
    issues: list[dict[str, str]] = []
    discovered_files: list[str] = []
    scanned_paths: list[str] = []

    if payload.input_directories:
        ingestion_report = ingestion_service.ingest_directories(
            [Path(path) for path in payload.input_directories],
            recursive=payload.recursive,
        )
        raw_documents.extend(ingestion_report.documents)
        scanned_paths = [str(path) for path in ingestion_report.scanned_paths]
        discovered_files = [str(path) for path in ingestion_report.discovered_files]
        issues.extend(
            {
                "path": str(issue.path),
                "reason": issue.reason,
                "detail": issue.detail,
            }
            for issue in ingestion_report.issues
        )

    raw_documents.extend(ingestion_service.ingest_document(document) for document in payload.documents)
    if not raw_documents:
        raise HTTPException(status_code=400, detail="No ingestible documents were produced from the request.")

    parsed_documents = parsing_service.parse_many(raw_documents)
    total_chunk_count = 0
    for parsed_document in parsed_documents:
        storage.save_parsed_document(parsed_document)
        chunks = chunking_service.chunk(parsed_document)
        storage.save_chunks(parsed_document.document_id, chunks)
        total_chunk_count += len(chunks)

    return IngestSourcesResponse(
        ingested_document_ids=[document.document_id for document in raw_documents],
        parsed_document_ids=[document.document_id for document in parsed_documents],
        chunk_count=total_chunk_count,
        issues=issues,
        trace_metadata={
            "request_kind": "ingest_sources",
            "filesystem_scan_used": bool(payload.input_directories),
            "inline_document_count": len(payload.documents),
            "scanned_paths": scanned_paths,
            "discovered_files": discovered_files,
        },
    )


@router.post("/index/refresh", response_model=IndexRefreshResponse)
def refresh_index(payload: IndexRefreshRequest) -> IndexRefreshResponse:
    """Build or refresh local vector and lexical manifests from persisted chunks.

    Why this function exists:
        Retrieval operates on persisted chunk manifests rather than raw
        documents. This endpoint gives the API a deterministic way to refresh a
        collection after ingestion and chunk persistence.

    Response structure:
        The response returns the refreshed collection name, chunk count,
        manifest paths, and trace metadata describing which indexing options
        were used.

    How the API maps to pipeline modules:
        The endpoint loads persisted chunks from ``LocalFileStorageService`` and
        passes them into ``IndexingService.prepare_indexes`` to refresh the
        vector and optional lexical manifests.
    """
    settings = get_settings()
    storage = LocalFileStorageService(settings)
    chunks = storage.load_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="No persisted chunks are available for indexing.")

    indexing_service = IndexingService(storage=storage, settings=settings)
    result = indexing_service.prepare_indexes(
        chunks,
        collection_name=payload.collection_name,
        include_lexical=payload.include_lexical,
    )

    return IndexRefreshResponse(
        collection_name=payload.collection_name,
        chunk_count=result.chunk_count,
        vector_manifest_path=str(result.vector_manifest_path),
        lexical_manifest_path=(
            str(result.lexical_manifest_path) if result.lexical_manifest_path is not None else None
        ),
        metadata_manifest_path=str(result.metadata_manifest_path),
        trace_metadata={
            "request_kind": "index_refresh",
            "include_lexical": payload.include_lexical,
            "source": "persisted_chunks",
        },
    )


@router.post("/retrieve", response_model=RetrievalExecutionResponse)
def retrieve_results(payload: RetrievalExecutionRequest) -> RetrievalExecutionResponse:
    """Run retrieval only and return ranked evidence with trace metadata.

    Why this function exists:
        Retrieval inspection is a core part of a grounded system. This endpoint
        lets callers inspect the evidence ranking independently from answer
        generation so debugging can isolate retrieval quality first.

    Response structure:
        The response returns the original query, the retrieved evidence list,
        and trace metadata such as collection name, hybrid mode, and result
        count.

    How the API maps to pipeline modules:
        The endpoint builds a ``RetrievalService`` from the named collection and
        executes the shared retrieval pipeline over vector and optional lexical
        indexes.
    """
    try:
        retrieval_service = RetrievalService.from_collection(
            collection_name=payload.collection_name,
            enable_hybrid=payload.enable_hybrid,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Requested collection has not been indexed yet.") from exc

    results = retrieval_service.retrieve(
        RetrievalRequest(
            query=payload.query,
            top_k=payload.top_k,
            source_types=payload.source_types,
            tag_filters=payload.tag_filters,
            include_archived=payload.include_archived,
            min_score=payload.min_score,
        )
    )

    return RetrievalExecutionResponse(
        query=payload.query,
        collection_name=payload.collection_name,
        results=results,
        trace_metadata={
            "request_kind": "retrieve",
            "enable_hybrid": payload.enable_hybrid,
            "result_count": len(results),
        },
    )


@router.post("/query", response_model=QueryExecutionResponse)
def submit_query(request: Request, payload: QueryExecutionRequest) -> QueryExecutionResponse:
    """Run deterministic orchestration, retrieval, and optional generation for a query.

    Why this function exists:
        This endpoint is the minimal end-to-end API entry point for the local
        demo and for future clients. It keeps the full request path inspectable
        by returning the orchestration plan, retrieval evidence, final answer,
        and trace metadata in one structured payload.

    Response structure:
        The response includes the selected task type, orchestration plan,
        retrieval results, optional generated answer, and trace metadata that
        records whether generation was attempted or skipped.

    How the API maps to pipeline modules:
        The endpoint first calls ``OrchestrationService`` to classify the query,
        then ``RetrievalService`` for evidence gathering, and finally
        ``GenerationService`` when the selected orchestration plan includes a
        generation step.
    """
    orchestration_service = _get_orchestration_service(request.app)
    generation_service = _get_generation_service(request.app)
    plan = orchestration_service.build_plan(OrchestrationRequest(user_request=payload.query))

    source_types = _extract_source_types_from_plan(plan)
    try:
        # Build the retrieval service from the requested collection before the
        # query runs so missing manifests produce a clear HTTP error instead of
        # an unstructured server exception.
        retrieval_service = RetrievalService.from_collection(collection_name=payload.collection_name)
    except FileNotFoundError as exc:
        # Surface the same collection-not-indexed behavior used by the
        # retrieval-only endpoint so callers can distinguish setup issues from
        # genuine empty retrieval results.
        raise HTTPException(status_code=404, detail="Requested collection has not been indexed yet.") from exc
    retrieval_results = retrieval_service.retrieve(
        RetrievalRequest(
            query=payload.query,
            top_k=payload.top_k,
            source_types=source_types,
            include_archived=payload.include_archived,
            min_score=payload.min_score,
        )
    )

    should_generate = any(step.component == "generation" for step in plan.steps)
    answer = None
    generation_error: str | None = None
    generation_fallback_used = False
    if should_generate and retrieval_results:
        try:
            generation_fallback_used = generation_service.uses_fallback_backend()
            answer = generation_service.answer(
                GenerationRequest(
                    query=payload.query,
                    task_type=plan.classification.task_type,
                    retrieved_context=retrieval_results,
                    output_instructions=payload.output_instructions,
                )
            )
        except Exception as exc:  # pragma: no cover - error path is represented in trace metadata
            generation_error = str(exc)
    elif should_generate and not retrieval_results:
        generation_error = "Generation skipped because retrieval returned no evidence."

    return QueryExecutionResponse(
        query=payload.query,
        task_type=plan.classification.task_type,
        orchestration_plan=plan,
        retrieval_results=retrieval_results,
        answer=answer,
        trace_metadata={
            "request_kind": "query",
            "collection_name": payload.collection_name,
            "generation_attempted": should_generate,
            "generation_succeeded": answer is not None and not generation_fallback_used,
            "generation_fallback_used": generation_fallback_used,
            "generation_backend_kind": generation_service.backend_kind(),
            "generation_model_name": answer.model_name if answer is not None else None,
            "generation_error": generation_error,
            "retrieval_result_count": len(retrieval_results),
        },
    )


@router.post("/evaluation/run", response_model=EvaluationExecutionResponse)
def run_evaluation(request: Request, payload: EvaluationExecutionRequest) -> EvaluationExecutionResponse:
    """Run the first-pass evaluation harness and optionally save its results.

    Why this function exists:
        The evaluation endpoint provides a deterministic way to benchmark the
        grounded generation stack through the same API surface used by local
        demos and future tooling.

    Response structure:
        The response returns per-case evaluation records, the saved report path
        when persistence is enabled, and trace metadata describing benchmark
        execution and local result saving.

    How the API maps to pipeline modules:
        The endpoint reuses the application's configured ``GenerationService``
        and passes the supplied test cases into ``EvaluationService``, which
        runs generation, validation, and optional local JSON persistence.
    """
    evaluation_service = EvaluationService(
        generation_service=_get_generation_service(request.app),
    )
    records = evaluation_service.run_benchmark(
        dataset_name=payload.dataset_name,
        test_cases=payload.test_cases,
        save_results=False,
    )
    saved_path = None
    if payload.save_results:
        saved_path = str(
            evaluation_service.save_results(
                dataset_name=payload.dataset_name,
                records=records,
            )
        )

    return EvaluationExecutionResponse(
        dataset_name=payload.dataset_name,
        records=records,
        saved_path=saved_path,
        trace_metadata={
            "request_kind": "evaluation_run",
            "case_count": len(payload.test_cases),
            "save_results": payload.save_results,
        },
    )


def demo_page() -> HTMLResponse:
    """Return a minimal local demo page for grounded query inspection.

    Why this function exists:
        The backend needs a lightweight inspectable interface that demonstrates
        the end-to-end grounded flow without introducing a separate frontend
        build toolchain. Returning a single HTML page keeps the demo easy to run
        locally and easy to audit in source control.

    Response structure:
        The page renders a small form and client-side script that call the API
        endpoints and display the final answer, citations, and retrieved
        evidence.

    How the API maps to pipeline modules:
        The page submits queries to the API's orchestrated query endpoint and
        renders the retrieval and generation outputs side by side so grounded
        evidence stays visible.
    """
    return HTMLResponse(_build_demo_html())


def _get_generation_service(application: FastAPI) -> GenerationService:
    """Return the generation service configured on the FastAPI application.

    Why this function exists:
        API handlers need a shared generation service instance so tests can
        override it through ``app.state`` and the demo page can reuse the same
        local backend configuration as the rest of the API.

    Response structure:
        This helper returns a service instance rather than an HTTP payload.

    How the API maps to pipeline modules:
        The helper resolves the ``GenerationService`` boundary used by the
        query and evaluation endpoints.
    """
    generation_service = getattr(application.state, "generation_service", None)
    if isinstance(generation_service, GenerationService):
        return generation_service
    return GenerationService()


def _get_orchestration_service(application: FastAPI) -> OrchestrationService:
    """Return the orchestration service configured on the FastAPI application.

    Why this function exists:
        The query endpoint needs access to the deterministic router and planner,
        and tests may want to override that service at the application level.

    Response structure:
        This helper returns a service instance rather than serialized JSON.

    How the API maps to pipeline modules:
        The helper resolves the ``OrchestrationService`` boundary that selects
        the task type and execution plan for query requests.
    """
    orchestration_service = getattr(application.state, "orchestration_service", None)
    if isinstance(orchestration_service, OrchestrationService):
        return orchestration_service
    return OrchestrationService()


def _extract_source_types_from_plan(plan: Any) -> list[str]:
    """Extract retrieval source filters recorded in the orchestration plan.

    Why this function exists:
        The deterministic planner may annotate retrieval steps with filters such
        as ``support_ticket`` only. This helper keeps the API's query endpoint
        aligned with the orchestration plan instead of duplicating task-specific
        routing logic a second time.

    Response structure:
        The helper returns a plain list of source types that can be passed into
        the retrieval request.

    How the API maps to pipeline modules:
        The helper bridges the orchestration plan into the retrieval module by
        translating plan-step metadata into retrieval filters.
    """
    for step in plan.steps:
        if step.component == "retrieval":
            source_types = step.inputs.get("source_types", [])
            if isinstance(source_types, list):
                return [str(source_type) for source_type in source_types]
    return []


def _build_demo_html() -> str:
    """Construct the inline HTML used for the minimal demo interface.

    Why this function exists:
        The project needs a no-build demo page that highlights grounded answers
        and retrieved evidence. Building the markup in one helper keeps the
        returned page easy to inspect and update without introducing a frontend
        build system.

    Response structure:
        The returned string contains the full HTML document served at the root
        path, including a small amount of inline JavaScript for API calls.

    How the API maps to pipeline modules:
        The page acts as a thin client over the API layer, which in turn maps
        onto orchestration, retrieval, and generation modules.
    """
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Local AgentOps Demo</title>
    <style>
      :root {
        --bg: #f6f2e9;
        --panel: #fffdf8;
        --ink: #1e1b16;
        --accent: #005f73;
        --line: #d8d0c2;
      }
      body {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        background: linear-gradient(180deg, #efe7da 0%, var(--bg) 100%);
        color: var(--ink);
      }
      main {
        max-width: 1100px;
        margin: 0 auto;
        padding: 32px 20px 64px;
      }
      h1, h2 {
        margin: 0 0 12px;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px;
        margin-top: 18px;
        box-shadow: 0 12px 30px rgba(30, 27, 22, 0.08);
      }
      textarea, input, button {
        font: inherit;
      }
      textarea, input {
        width: 100%;
        box-sizing: border-box;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 12px;
        background: #fff;
      }
      textarea {
        min-height: 120px;
        resize: vertical;
      }
      button {
        background: var(--accent);
        color: white;
        border: 0;
        border-radius: 999px;
        padding: 10px 18px;
        cursor: pointer;
      }
      pre {
        white-space: pre-wrap;
        word-break: break-word;
        background: #f8f5ef;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid var(--line);
      }
      .grid {
        display: grid;
        gap: 18px;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      }
      .citation {
        border-top: 1px solid var(--line);
        padding-top: 10px;
        margin-top: 10px;
      }
      .meta {
        color: #5d564c;
        font-size: 0.95rem;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>Grounded Local Answer Demo</h1>
      <p class="meta">Submit a query, inspect retrieved evidence, then inspect the final answer and citations returned by the API.</p>
      <section class="panel">
        <label for="collection">Collection</label>
        <input id="collection" value="default" />
        <label for="query" style="display:block; margin-top:14px;">Query</label>
        <textarea id="query">How do I reset my password?</textarea>
        <div style="margin-top:14px;">
          <button id="run-query">Run Grounded Query</button>
        </div>
      </section>
      <div class="grid">
        <section class="panel">
          <h2>Final Answer</h2>
          <pre id="answer">No answer yet.</pre>
          <div id="citations"></div>
        </section>
        <section class="panel">
          <h2>Retrieved Evidence</h2>
          <div id="evidence">No evidence yet.</div>
        </section>
      </div>
      <section class="panel">
        <h2>Trace Metadata</h2>
        <pre id="trace">{}</pre>
      </section>
      <section class="panel">
        <h2>Backend Status</h2>
        <pre id="status">Idle.</pre>
      </section>
    </main>
    <script>
      const queryButton = document.getElementById("run-query");
      queryButton.addEventListener("click", async () => {
        const query = document.getElementById("query").value;
        const collection = document.getElementById("collection").value;
        document.getElementById("status").textContent = "Running query...";

        try {
          const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, collection_name: collection, top_k: 5 })
          });
          const payload = await response.json();

          if (!response.ok) {
            const errorDetail = payload.detail || "Unknown API error.";
            document.getElementById("answer").textContent = "No answer generated.";
            document.getElementById("citations").innerHTML = "<p>No citations returned.</p>";
            document.getElementById("evidence").innerHTML = "No evidence returned.";
            document.getElementById("trace").textContent = JSON.stringify({ http_status: response.status, detail: errorDetail }, null, 2);
            document.getElementById("status").textContent = `API error (${response.status}): ${errorDetail}`;
            return;
          }

          document.getElementById("answer").textContent = payload.answer ? payload.answer.answer : "No answer generated.";
          document.getElementById("trace").textContent = JSON.stringify(payload.trace_metadata, null, 2);

          const evidenceHtml = (payload.retrieval_results || []).map((result) => `
            <div class="citation">
              <strong>${result.title}</strong><br />
              <span class="meta">${result.chunk_id} • ${result.citation.locator} • score ${result.score.toFixed(3)}</span>
              <p>${result.text}</p>
            </div>
          `).join("");
          document.getElementById("evidence").innerHTML = evidenceHtml || "No evidence returned.";

          const citationHtml = payload.answer && payload.answer.citations
            ? payload.answer.citations.map((citation) => `
                <div class="citation">
                  <strong>${citation.title}</strong><br />
                  <span class="meta">${citation.chunk_id} • ${citation.locator}</span>
                  <p>${citation.snippet}</p>
                </div>
              `).join("")
            : "<p>No citations returned.</p>";
          document.getElementById("citations").innerHTML = citationHtml;

          const trace = payload.trace_metadata || {};
          const backendStatus = [
            `Generation attempted: ${Boolean(trace.generation_attempted)}`,
            `Generation succeeded: ${Boolean(trace.generation_succeeded)}`,
            `Fallback used: ${Boolean(trace.generation_fallback_used)}`,
            `Backend kind: ${trace.generation_backend_kind || "n/a"}`,
            `Model name: ${trace.generation_model_name || "n/a"}`,
            `Generation error: ${trace.generation_error || "none"}`
          ].join("\\n");
          document.getElementById("status").textContent = backendStatus;
        } catch (error) {
          document.getElementById("answer").textContent = "No answer generated.";
          document.getElementById("citations").innerHTML = "<p>No citations returned.</p>";
          document.getElementById("evidence").innerHTML = "No evidence returned.";
          document.getElementById("trace").textContent = JSON.stringify({ client_error: String(error) }, null, 2);
          document.getElementById("status").textContent = `Client error: ${error}`;
        }
      });
    </script>
  </body>
</html>"""
