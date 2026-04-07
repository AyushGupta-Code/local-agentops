"""API tests for the main local pipeline endpoints."""

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.generation.backends import LocalModelService, StaticJSONLocalModelBackend
from app.generation.service import GenerationService
from app.main import create_application


def _build_client(tmp_path, monkeypatch) -> TestClient:
    """Create a FastAPI test client rooted in a temporary data directory.

    Why this function exists:
        The API endpoints persist processed chunks, indexes, and evaluation
        output to managed local directories. Tests need those paths isolated per
        run so endpoint behavior can be exercised end to end without touching
        repository state.

    Parameters:
        tmp_path: Pytest temporary directory used as the managed data root.
        monkeypatch: Pytest monkeypatch fixture used to override environment
            variables before app creation.

    Returns:
        A ``TestClient`` bound to a newly created FastAPI application.

    Edge cases handled:
        The cached settings object is cleared before app creation so the app
        reads the temporary test paths instead of any previously cached values.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("RAW_DATA_DIR", "raw")
    monkeypatch.setenv("PROCESSED_DATA_DIR", "processed")
    monkeypatch.setenv("INDEX_DATA_DIR", "indexes")
    monkeypatch.setenv("EVAL_DATA_DIR", "eval")
    monkeypatch.setenv("LOCAL_LLM_PROVIDER", "custom")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "fixture-model")
    get_settings.cache_clear()
    return TestClient(create_application())


def test_api_endpoints_support_ingest_index_retrieve_query_and_evaluation(tmp_path, monkeypatch) -> None:
    """Verify that the main API endpoints cooperate across the local pipeline."""
    client = _build_client(tmp_path, monkeypatch)

    # Ingest one inline document because the end-to-end API flow starts by
    # persisting parsed documents and chunks for later indexing.
    ingest_response = client.post(
        "/api/ingest/sources",
        json={
            "documents": [
                {
                    "document_id": "doc-api-1",
                    "source_type": "document",
                    "connector_name": "api-test",
                    "title": "Password Reset Guide",
                    "content": "Reset your password from the login page. If the email is missing, check the spam folder.",
                    "mime_type": "text/plain",
                    "source_uri": "kb://password-reset",
                    "metadata": {"section_title": "Reset Steps"},
                }
            ]
        },
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunk_count"] >= 1

    # Refresh the local collection because retrieval endpoints depend on the
    # manifests produced by the indexing pipeline.
    index_response = client.post(
        "/api/index/refresh",
        json={"collection_name": "demo", "include_lexical": True},
    )
    assert index_response.status_code == 200
    assert index_response.json()["collection_name"] == "demo"

    # Run retrieval first so the test can capture a valid chunk id for the
    # mocked generation backend used in the end-to-end query path.
    retrieve_response = client.post(
        "/api/retrieve",
        json={"query": "How do I reset my password?", "collection_name": "demo", "top_k": 3},
    )
    assert retrieve_response.status_code == 200
    retrieval_payload = retrieve_response.json()
    assert retrieval_payload["results"]
    first_chunk_id = retrieval_payload["results"][0]["chunk_id"]

    # Override the app's generation service with a deterministic backend so the
    # query and evaluation endpoints can exercise grounded answer assembly.
    client.app.state.generation_service = GenerationService(
        local_model_service=LocalModelService(
            backend=StaticJSONLocalModelBackend(
                {
                    "answer": "Use the reset link from the login page. If the email is missing, check the spam folder.",
                    "cited_chunk_ids": [first_chunk_id],
                    "confidence_notes": [],
                    "follow_up_questions": [],
                },
                model_name="api-test-backend",
            )
        ),
        settings=get_settings(),
    )

    # Submit the full end-to-end query because the demo interface depends on
    # the orchestrated response containing both retrieval evidence and answer.
    query_response = client.post(
        "/api/query",
        json={"query": "How do I reset my password?", "collection_name": "demo", "top_k": 3},
    )
    assert query_response.status_code == 200
    query_payload = query_response.json()
    assert query_payload["task_type"] == "document_qa"
    assert query_payload["answer"]["citations"][0]["chunk_id"] == first_chunk_id
    assert query_payload["trace_metadata"]["generation_succeeded"] is True
    assert query_payload["trace_metadata"]["generation_fallback_used"] is False
    assert query_payload["trace_metadata"]["generation_backend_kind"] == "static_json"

    # Run the evaluation harness through the API because the benchmark endpoint
    # should record validation outcomes and save auditable local output.
    evaluation_response = client.post(
        "/api/evaluation/run",
        json={
            "dataset_name": "api-smoke",
            "save_results": True,
            "test_cases": [
                {
                    "evaluation_id": "case-1",
                    "query": "How do I reset my password?",
                    "task_type": "document_qa",
                    "retrieved_context": retrieval_payload["results"],
                }
            ],
        },
    )
    assert evaluation_response.status_code == 200
    evaluation_payload = evaluation_response.json()
    assert evaluation_payload["records"][0]["validation_outcomes"]
    assert evaluation_payload["saved_path"].endswith("api-smoke/latest_results.json")


def test_demo_page_is_served_at_root(tmp_path, monkeypatch) -> None:
    """Verify that the minimal demo interface is available from the root path."""
    client = _build_client(tmp_path, monkeypatch)

    response = client.get("/")

    assert response.status_code == 200
    assert "Grounded Local Answer Demo" in response.text
    assert "Backend Status" in response.text


def test_query_trace_marks_fallback_answers_as_not_true_generation_success(tmp_path, monkeypatch) -> None:
    """Verify that fallback answers are distinguished from real backend success."""
    client = _build_client(tmp_path, monkeypatch)

    # Seed one simple document so retrieval can succeed and the query endpoint
    # reaches the generation layer using the default fallback backend.
    ingest_response = client.post(
        "/api/ingest/sources",
        json={
            "documents": [
                {
                    "document_id": "doc-api-fallback",
                    "source_type": "document",
                    "connector_name": "api-test",
                    "title": "Password Reset Guide",
                    "content": "Reset your password from the login page. If the email is missing, check the spam folder.",
                    "mime_type": "text/plain",
                    "source_uri": "kb://password-reset",
                    "metadata": {"section_title": "Reset Steps"},
                }
            ]
        },
    )
    assert ingest_response.status_code == 200

    index_response = client.post(
        "/api/index/refresh",
        json={"collection_name": "demo", "include_lexical": True},
    )
    assert index_response.status_code == 200

    # Submit the query without overriding the app's generation service so the
    # default fallback backend path is exercised.
    query_response = client.post(
        "/api/query",
        json={"query": "How do I reset my password?", "collection_name": "demo", "top_k": 3},
    )
    assert query_response.status_code == 200
    payload = query_response.json()

    assert payload["answer"]["answer"] == "Local model backend not configured."
    assert payload["trace_metadata"]["generation_attempted"] is True
    assert payload["trace_metadata"]["generation_succeeded"] is False
    assert payload["trace_metadata"]["generation_fallback_used"] is True
    assert payload["trace_metadata"]["generation_backend_kind"] == "fallback"


def test_query_returns_404_for_missing_collection(tmp_path, monkeypatch) -> None:
    """Verify that the query endpoint reports unindexed collections cleanly."""
    client = _build_client(tmp_path, monkeypatch)

    # Submit a query before any indexing occurs because the endpoint should
    # surface a clean not-found error rather than an internal server error.
    response = client.post(
        "/api/query",
        json={"query": "How do I reset my password?", "collection_name": "missing", "top_k": 3},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Requested collection has not been indexed yet."
