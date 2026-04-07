"""Abstractions and local implementations for model-backed generation."""

import json
from typing import Any, Callable, Protocol
from urllib import error, request

from app.core.config import Settings, get_settings


class LocalModelBackend(Protocol):
    """Protocol describing the generation call surface used by the app."""

    model_name: str
    backend_kind: str
    is_fallback_backend: bool

    def generate(
        self,
        *,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        """Generate a model response string from a fully assembled prompt.

        Why this function exists:
            The generation service should depend on a minimal backend interface
            so prompt assembly, answer validation, and local-model invocation
            can evolve independently. The backend only needs to accept the final
            prompt and generation controls, then return raw model text.

        Parameters:
            prompt: Fully assembled grounded prompt.
            temperature: Sampling temperature requested by the caller.
            max_output_tokens: Upper bound on generated output length.

        Returns:
            Raw model output text, usually JSON for this generation pipeline.

        Edge cases handled:
            The backend contract intentionally returns plain text because
            different local providers expose different response shapes; parsing
            and validation remain the responsibility of the generation service.
        """


class StaticJSONLocalModelBackend:
    """Deterministic backend used for tests and local sample scripts."""

    def __init__(self, response_payload: dict[str, Any], model_name: str = "static-json-backend") -> None:
        """Store a fixed JSON payload that will be returned for every generation call.

        Why this function exists:
            Tests for prompt construction and answer assembly should not depend
            on a running local model. This backend provides a deterministic way
            to exercise the generation pipeline end to end.

        Parameters:
            response_payload: JSON-serializable model response returned on every
                call.
            model_name: Human-readable backend name surfaced in generation
                responses.

        Returns:
            This initializer stores the static payload and backend name.

        Edge cases handled:
            Because the payload is serialized on every call, tests exercise the
            same JSON decoding path used for real model backends.
        """
        self._response_payload = dict(response_payload)
        self.model_name = model_name
        self.backend_kind = "static_json"
        self.is_fallback_backend = False

    def generate(
        self,
        *,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        """Return the configured JSON payload regardless of generation settings.

        Why this function exists:
            The generation pipeline still needs to pass through a backend call
            in tests and sample scripts. Returning a fixed payload keeps those
            tests deterministic while preserving the same service layering used
            with real local models.

        Parameters:
            prompt: Fully assembled prompt. Accepted for API compatibility.
            temperature: Requested sampling temperature. Accepted for API
                compatibility.
            max_output_tokens: Requested output-token budget. Accepted for API
                compatibility.

        Returns:
            A JSON string built from the configured static payload.

        Edge cases handled:
            The method ignores prompt and sampling controls intentionally so the
            caller can validate downstream answer assembly independently from
            model nondeterminism.
        """
        _ = prompt
        _ = temperature
        _ = max_output_tokens
        return json.dumps(self._response_payload)


class FallbackLocalModelBackend(StaticJSONLocalModelBackend):
    """Explicit fallback backend used when no real local model adapter is configured."""

    def __init__(self, response_payload: dict[str, Any], model_name: str) -> None:
        """Mark the static backend as a non-real fallback answer source.

        Why this function exists:
            The API should distinguish between a real local-model answer and a
            deterministic placeholder response. This subclass preserves the same
            JSON behavior while making the fallback state explicit.

        Parameters:
            response_payload: JSON payload returned for fallback generation.
            model_name: Human-readable provider/model label surfaced in traces.

        Returns:
            This initializer stores the fallback payload and backend metadata.

        Edge cases handled:
            The fallback backend still returns valid JSON so downstream answer
            assembly can run and the UI can display a deterministic placeholder.
        """
        super().__init__(response_payload=response_payload, model_name=model_name)
        self.backend_kind = "fallback"
        self.is_fallback_backend = True


class OllamaLocalModelBackend:
    """Local model backend that calls an Ollama HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout_seconds: int,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        """Store the connection details required for Ollama generation calls.

        Why this function exists:
            Ollama is the first real local backend supported by the project.
            Keeping it behind the shared backend interface lets the rest of the
            generation stack stay unaware of provider-specific HTTP details.

        Parameters:
            base_url: Base URL of the local Ollama server.
            model_name: Concrete model name to send in each generation request.
            timeout_seconds: Network timeout for the local HTTP call.
            urlopen: Optional HTTP opener override used by tests.

        Returns:
            This initializer stores the backend configuration on the instance.

        Edge cases handled:
            The base URL is normalized to avoid duplicate slashes, and the HTTP
            opener is injectable so tests do not need a live Ollama server.
        """
        self._base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.backend_kind = "ollama"
        self.is_fallback_backend = False
        self._timeout_seconds = timeout_seconds
        self._urlopen = urlopen or request.urlopen

    def generate(
        self,
        *,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        """Send a grounded prompt to Ollama and return the raw response text.

        Why this function exists:
            The generation service needs a real local-model execution path.
            This method translates the shared generation contract into the
            Ollama HTTP API shape while keeping prompt assembly separate.

        Parameters:
            prompt: Fully assembled grounded prompt to send to Ollama.
            temperature: Sampling temperature forwarded to Ollama options.
            max_output_tokens: Maximum output token budget forwarded as
                ``num_predict``.

        Returns:
            The raw response string returned by Ollama, expected to contain the
            JSON answer contract requested by prompt assembly.

        Edge cases handled:
            HTTP connection failures, non-JSON responses, and missing ``response``
            fields all raise explicit ``RuntimeError`` instances so generation
            failures are surfaced rather than hidden.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
        }
        request_payload = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self._base_url}/api/generate",
            data=request_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            # Execute the local HTTP request against Ollama because generation
            # should now flow through a real provider when configured.
            with self._urlopen(http_request, timeout=self._timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"Ollama generation request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc

        model_response = response_payload.get("response")
        if not isinstance(model_response, str) or not model_response.strip():
            raise RuntimeError("Ollama response did not include a non-empty 'response' field.")

        return model_response


class LocalModelService:
    """Small adapter that exposes the selected local model provider through one interface."""

    def __init__(
        self,
        *,
        backend: LocalModelBackend | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Configure the model service with a concrete backend or a safe default.

        Why this function exists:
            The project needs a service-level abstraction for local generation
            that is separate from prompt assembly and answer validation. This
            adapter makes that split explicit and gives the rest of the
            generation pipeline one stable place to call.

        Parameters:
            backend: Concrete local model backend implementation to use.
            settings: Optional settings object used for provider and model
                metadata when no backend is explicitly supplied.

        Returns:
            This initializer stores the backend and settings on the service.

        Edge cases handled:
            When no backend is supplied, the service falls back to a static JSON
            backend that explicitly reports the local model as unimplemented
            rather than pretending a real provider call occurred.
        """
        self._settings = settings or get_settings()
        self._backend = backend or self._build_default_backend()

    def generate(
        self,
        *,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        """Delegate prompt execution to the configured local model backend.

        Why this function exists:
            The rest of the generation layer should not know which concrete
            local backend is in use. This method isolates that call boundary so
            future providers such as Ollama or llama.cpp can be added without
            reshaping prompt assembly or answer validation.

        Parameters:
            prompt: Fully assembled prompt to send to the local model.
            temperature: Requested temperature value.
            max_output_tokens: Maximum output-token budget for the response.

        Returns:
            Raw backend response text.

        Edge cases handled:
            The service performs no output parsing here because provider output
            formats can vary; validation stays in the answer assembly layer.
        """
        return self._backend.generate(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def backend_kind(self) -> str:
        """Return the provider/backend kind associated with the active backend.

        Why this function exists:
            Trace metadata should distinguish real provider execution from the
            fallback path. This helper exposes that backend classification
            without leaking backend implementation details to callers.

        Parameters:
            This helper takes no parameters because it reflects the backend
            already configured on the service.

        Returns:
            A short backend kind label such as ``ollama`` or ``fallback``.

        Edge cases handled:
            Backends without an explicit ``backend_kind`` attribute fall back to
            ``unknown`` so trace generation never fails due to missing metadata.
        """
        backend_kind = getattr(self._backend, "backend_kind", None)
        if isinstance(backend_kind, str) and backend_kind.strip():
            return backend_kind
        return "unknown"

    def uses_fallback_backend(self) -> bool:
        """Report whether the active backend is only a deterministic fallback.

        Why this function exists:
            The API needs to tell the difference between a real local-model
            answer and a placeholder response. This helper makes that
            distinction explicit in one place.

        Parameters:
            This helper takes no parameters because it inspects the configured
            backend already stored on the service.

        Returns:
            ``True`` when the current backend is a deterministic fallback rather
            than a live local-model adapter.

        Edge cases handled:
            Backends without an explicit flag default to ``False`` so existing
            custom test backends continue to behave like real backends.
        """
        return bool(getattr(self._backend, "is_fallback_backend", False))

    def model_name(self) -> str:
        """Return the human-readable model name associated with the active backend.

        Why this function exists:
            Generation responses should record which local backend produced the
            answer for debugging and evaluation. This helper centralizes that
            metadata lookup instead of requiring callers to inspect backend
            internals directly.

        Parameters:
            This helper takes no parameters because it reflects the active
            backend already stored on the service.

        Returns:
            A model/backend name string for the current local generation path.

        Edge cases handled:
            Backends without an explicit ``model_name`` attribute fall back to
            the settings-derived provider and model name so response metadata is
            never blank.
        """
        backend_model_name = getattr(self._backend, "model_name", None)
        if isinstance(backend_model_name, str) and backend_model_name.strip():
            return backend_model_name

        return f"{self._settings.local_llm_provider}:{self._settings.local_llm_model}"

    def _build_default_backend(self) -> LocalModelBackend:
        """Select the default backend from application settings.

        Why this function exists:
            The generation layer should prefer a real provider when one is
            implemented in configuration, but it still needs a safe fallback for
            unsupported providers or unconfigured local environments.

        Parameters:
            This helper takes no explicit parameters because it reads the
            already-resolved settings stored on the service.

        Returns:
            A concrete backend implementing the shared local-model interface.

        Edge cases handled:
            Unsupported providers fall back to the deterministic backend rather
            than pretending a real adapter exists.
        """
        if self._settings.local_llm_provider == "ollama":
            return OllamaLocalModelBackend(
                base_url=self._settings.local_llm_base_url,
                model_name=self._settings.local_llm_model,
                timeout_seconds=self._settings.local_llm_timeout_seconds,
            )

        return FallbackLocalModelBackend(
            {
                "answer": "Local model backend not configured.",
                "cited_chunk_ids": [],
                "confidence_notes": ["No concrete local model backend is configured."],
                "follow_up_questions": [],
            },
            model_name=f"{self._settings.local_llm_provider}:{self._settings.local_llm_model}",
        )
