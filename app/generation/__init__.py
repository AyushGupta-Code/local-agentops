"""Generation package for citation-grounded answer synthesis."""

from app.generation.backends import (
    FallbackLocalModelBackend,
    LocalModelBackend,
    LocalModelService,
    OllamaLocalModelBackend,
    StaticJSONLocalModelBackend,
)
from app.generation.prompting import PromptAssemblyService
from app.generation.service import AnswerAssemblyService, GenerationService

__all__ = [
    "AnswerAssemblyService",
    "FallbackLocalModelBackend",
    "GenerationService",
    "LocalModelBackend",
    "LocalModelService",
    "OllamaLocalModelBackend",
    "PromptAssemblyService",
    "StaticJSONLocalModelBackend",
]
