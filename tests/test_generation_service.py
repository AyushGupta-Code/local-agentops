"""Unit tests for prompt assembly and grounded local generation."""

import json
from urllib import request

import pytest

from app.core.config import Settings
from app.generation.backends import (
    LocalModelService,
    OllamaLocalModelBackend,
    StaticJSONLocalModelBackend,
)
from app.generation.prompting import PromptAssemblyService
from app.generation.service import GenerationService
from app.models.schemas import Citation, GenerationRequest, RetrievalResult


def _build_retrieval_results() -> list[RetrievalResult]:
    """Create small grounded retrieval fixtures for generation tests.

    Why this function exists:
        Generation tests need concrete retrieval results with citation metadata
        because answer assembly is required to preserve citations rather than
        invent them. Keeping the fixture builder in one helper avoids repeating
        verbose schema construction in every test.

    Parameters:
        This helper takes no parameters because the tests all use the same small
        grounded evidence set.

    Returns:
        A list of retrieval results suitable for prompt assembly and answer
        validation tests.

    Edge cases handled:
        The fixtures include multiple chunks so tests can verify that cited
        chunk ids are resolved against a closed evidence set rather than a
        single hard-coded citation.
    """
    return [
        RetrievalResult(
            chunk_id="doc-1:0",
            document_id="doc-1",
            source_type="document",
            score=0.91,
            rank=1,
            text="Reset your password from the login page by using the emailed reset link.",
            title="Password Reset Guide",
            section_heading="Reset Steps",
            source_uri="kb://password-reset",
            document_metadata={"source_uri": "kb://password-reset"},
            metadata={"vector_score": 0.9, "lexical_score": 0.8},
            provenance={"retrieval_mode": "hybrid"},
            citation=Citation(
                chunk_id="doc-1:0",
                document_id="doc-1",
                source_type="document",
                title="Password Reset Guide",
                locator="Section: Reset Steps",
                snippet="Reset your password from the login page by using the emailed reset link.",
                source_uri="kb://password-reset",
            ),
        ),
        RetrievalResult(
            chunk_id="doc-1:1",
            document_id="doc-1",
            source_type="document",
            score=0.74,
            rank=2,
            text="If the reset email does not arrive, verify the account email address and spam folder.",
            title="Password Reset Guide",
            section_heading="Troubleshooting",
            source_uri="kb://password-reset",
            document_metadata={"source_uri": "kb://password-reset"},
            metadata={"vector_score": 0.7, "lexical_score": 0.6},
            provenance={"retrieval_mode": "hybrid"},
            citation=Citation(
                chunk_id="doc-1:1",
                document_id="doc-1",
                source_type="document",
                title="Password Reset Guide",
                locator="Section: Troubleshooting",
                snippet="If the reset email does not arrive, verify the account email address and spam folder.",
                source_uri="kb://password-reset",
            ),
        ),
    ]


def test_prompt_assembly_includes_task_question_and_citation_metadata() -> None:
    """Verify that prompt assembly keeps evidence and citation tracking explicit."""
    prompt_service = PromptAssemblyService()
    request = GenerationRequest(
        query="How do I reset my password?",
        task_type="document_qa",
        retrieved_context=_build_retrieval_results(),
        output_instructions=["Keep the answer concise."],
    )

    # Assemble the prompt because the generation layer must expose task type,
    # evidence ids, locator metadata, and output instructions explicitly.
    prompt = prompt_service.assemble_prompt(request)

    assert "Task type: document_qa" in prompt
    assert "User question:\nHow do I reset my password?" in prompt
    assert "Chunk id: doc-1:0" in prompt
    assert "Locator: Section: Reset Steps" in prompt
    assert "Keep the answer concise." in prompt
    assert '"cited_chunk_ids": [string]' in prompt


def test_generation_service_preserves_retrieved_citations_in_answer() -> None:
    """Verify that answer assembly reuses citations from retrieval results only."""
    backend = StaticJSONLocalModelBackend(
        {
            "answer": "Use the reset link from the login page. If the email does not arrive, check the spam folder.",
            "cited_chunk_ids": ["doc-1:0", "doc-1:1"],
            "confidence_notes": [],
            "follow_up_questions": ["Do you want troubleshooting steps for expired links?"],
        },
        model_name="mock-local-model",
    )
    generation_service = GenerationService(
        local_model_service=LocalModelService(backend=backend),
    )

    # Run generation with a mocked backend because the test should exercise the
    # full prompt -> model -> answer assembly path without a real local model.
    response = generation_service.answer(
        GenerationRequest(
            query="How do I reset my password?",
            task_type="document_qa",
            retrieved_context=_build_retrieval_results(),
        )
    )

    assert response.answer.startswith("Use the reset link")
    assert response.citations[0].chunk_id == "doc-1:0"
    assert response.citations[1].locator == "Section: Troubleshooting"
    assert response.used_chunk_ids == ["doc-1:0", "doc-1:1"]
    assert response.supporting_evidence_summary[0].startswith("Password Reset Guide")
    assert response.model_name == "mock-local-model"


def test_generation_service_raises_on_unknown_cited_chunk_ids() -> None:
    """Verify that the generation layer fails closed on invented citations."""
    backend = StaticJSONLocalModelBackend(
        {
            "answer": "Use the reset link from the login page.",
            "cited_chunk_ids": ["missing-chunk"],
            "confidence_notes": ["Evidence may be incomplete."],
            "follow_up_questions": [],
        }
    )
    generation_service = GenerationService(
        local_model_service=LocalModelService(backend=backend),
    )

    # Attempt to assemble an answer with an unknown chunk id because the
    # generation layer must not silently invent citations.
    with pytest.raises(ValueError, match="unknown chunk id"):
        generation_service.answer(
            GenerationRequest(
                query="How do I reset my password?",
                task_type="document_qa",
                retrieved_context=_build_retrieval_results(),
            )
        )


def test_generation_service_returns_confidence_notes_and_heuristic_confidence() -> None:
    """Verify that uncertainty notes are preserved and influence confidence."""
    backend = StaticJSONLocalModelBackend(
        {
            "answer": "The evidence suggests checking the spam folder if the reset email does not arrive.",
            "cited_chunk_ids": ["doc-1:1"],
            "confidence_notes": ["Only troubleshooting evidence was retrieved."],
            "follow_up_questions": [],
        }
    )
    generation_service = GenerationService(
        local_model_service=LocalModelService(backend=backend),
    )

    response = generation_service.answer(
        GenerationRequest(
            query="What if the password reset email is missing?",
            task_type="policy_lookup",
            retrieved_context=_build_retrieval_results(),
        )
    )

    assert response.confidence_notes == ["Only troubleshooting evidence was retrieved."]
    assert response.confidence is not None
    assert response.confidence < 0.74


def test_local_model_service_marks_explicit_static_backend_as_non_fallback() -> None:
    """Verify that explicit test backends are not reported as fallback generation."""
    backend = StaticJSONLocalModelBackend(
        {
            "answer": "Use the reset link.",
            "cited_chunk_ids": [],
            "confidence_notes": [],
            "follow_up_questions": [],
        },
        model_name="explicit-test-backend",
    )
    service = LocalModelService(backend=backend)

    # Assert that an explicitly injected backend is treated as a real backend
    # path for trace-metadata purposes even if it is deterministic in tests.
    assert service.uses_fallback_backend() is False
    assert service.backend_kind() == "static_json"


def test_local_model_service_uses_fallback_for_unsupported_default_provider() -> None:
    """Verify that unsupported providers fall back explicitly rather than pretending to run."""
    settings = Settings(
        LOCAL_LLM_PROVIDER="custom",
        LOCAL_LLM_MODEL="demo-model",
    )
    service = LocalModelService(settings=settings)

    # Assert that the default-provider path is marked as a fallback when no
    # real backend adapter exists for the configured provider.
    assert service.uses_fallback_backend() is True
    assert service.backend_kind() == "fallback"


def test_ollama_backend_formats_http_request_and_returns_response_text() -> None:
    """Verify that the Ollama backend sends the expected request payload."""

    class _FakeResponse:
        """Tiny response wrapper used to simulate an HTTP response body."""

        def __enter__(self):
            """Return the response object itself for context-manager usage."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Propagate any exception raised inside the context manager."""
            return False

        def read(self) -> bytes:
            """Return a minimal Ollama-style JSON payload as bytes."""
            return json.dumps({"response": "{\"answer\": \"ok\", \"cited_chunk_ids\": [], \"confidence_notes\": [], \"follow_up_questions\": []}"}).encode("utf-8")

    captured_request: dict[str, object] = {}

    def _fake_urlopen(http_request: request.Request, timeout: int):
        """Capture the outgoing request so the test can inspect its payload."""
        captured_request["url"] = http_request.full_url
        captured_request["body"] = json.loads(http_request.data.decode("utf-8"))
        captured_request["timeout"] = timeout
        return _FakeResponse()

    backend = OllamaLocalModelBackend(
        base_url="http://127.0.0.1:11434",
        model_name="llama3.1",
        timeout_seconds=30,
        urlopen=_fake_urlopen,
    )

    # Execute one generation call so the backend has to construct the real HTTP
    # request payload expected by Ollama.
    output = backend.generate(
        prompt="Prompt text",
        temperature=0.2,
        max_output_tokens=256,
    )

    assert captured_request["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured_request["body"]["model"] == "llama3.1"
    assert captured_request["body"]["options"]["num_predict"] == 256
    assert captured_request["timeout"] == 30
    assert '"answer": "ok"' in output
