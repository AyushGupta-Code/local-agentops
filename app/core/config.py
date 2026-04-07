"""Environment-backed settings and path helpers for the Local AgentOps application."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_repo_root() -> Path:
    """Return the repository root used for resolving relative data paths.

    Why this function exists:
        The application stores most local state under repository-managed
        directories such as ``data/raw`` and ``data/indexes``. Centralizing the
        repo-root calculation keeps path resolution consistent across API
        startup, ingestion jobs, and future offline scripts.

    Parameters:
        This helper does not take parameters because the repository root is
        derived from the location of this source file.

    Returns:
        A fully resolved ``Path`` pointing at the repository root directory.

    Edge cases handled:
        The function resolves symlinks and relative components so callers get a
        stable absolute path even when the project is launched from another
        working directory.
    """
    # Walk upward from this module to the repository root so path resolution is
    # independent from the shell's current working directory.
    repo_root = Path(__file__).resolve().parents[2]

    # Return the absolute repository root so all downstream helpers can anchor
    # paths against a single trusted base location.
    return repo_root


def _normalize_local_path(path_value: Path | str, *, base_path: Path) -> Path:
    """Resolve a user-provided local path against a trusted base directory.

    Why this function exists:
        Configuration values may be relative or absolute. The shared settings
        layer needs a single normalization routine so every data directory is
        converted into an absolute path before it is used by ingestion,
        indexing, or evaluation code.

    Parameters:
        path_value: Candidate filesystem location supplied by defaults or
            environment variables.
        base_path: Trusted parent directory used when ``path_value`` is
            relative.

    Returns:
        An absolute ``Path`` representing the normalized location.

    Edge cases handled:
        Relative paths are anchored to ``base_path``; absolute paths are
        preserved; redundant ``..`` or ``.`` segments are collapsed by
        ``resolve(strict=False)`` without requiring the directory to exist.
    """
    # Convert string input into a ``Path`` so resolution behavior is uniform.
    candidate_path = Path(path_value)

    # Anchor relative paths under the trusted base directory while leaving
    # already-absolute paths unchanged.
    if not candidate_path.is_absolute():
        candidate_path = base_path / candidate_path

    # Resolve the path without requiring it to exist yet because the initial
    # scaffold may run before any data directories are created.
    normalized_path = candidate_path.resolve(strict=False)

    # Return the canonical absolute path for later validation and use.
    return normalized_path


def _assert_within_directory(path_value: Path, *, parent_directory: Path, field_name: str) -> Path:
    """Validate that a path stays inside a trusted parent directory.

    Why this function exists:
        Environment variables are user-controlled input. This helper prevents
        accidental or malicious configuration that points runtime-managed data
        directories outside the intended repository storage area.

    Parameters:
        path_value: Absolute path that has already been normalized.
        parent_directory: Trusted directory that must contain ``path_value``.
        field_name: Human-readable config field name used in validation errors.

    Returns:
        The original ``path_value`` when validation succeeds so callers can
        chain the helper inside config validators.

    Edge cases handled:
        The function allows the child path to equal the parent path, and it
        raises a clear ``ValueError`` when the candidate escapes via ``..`` or
        by pointing to another absolute location.
    """
    # Confirm the candidate lives under the allowed directory to prevent path
    # traversal or accidental writes into unrelated parts of the filesystem.
    if path_value != parent_directory and parent_directory not in path_value.parents:
        raise ValueError(f"{field_name} must stay within {parent_directory}")

    # Return the validated path unchanged so callers can keep using it.
    return path_value


def resolve_data_subdirectory(data_root: Path, configured_path: Path | str, field_name: str) -> Path:
    """Resolve and validate a data subdirectory under the configured data root.

    Why this function exists:
        Multiple settings fields represent directories under ``data_root``.
        This helper keeps their normalization and safety checks consistent so
        future modules can trust that configured storage locations stay local.

    Parameters:
        data_root: Trusted absolute root for all managed application data.
        configured_path: Directory value from settings defaults or environment.
        field_name: Name of the config field currently being validated.

    Returns:
        An absolute, validated path under ``data_root``.

    Edge cases handled:
        Relative directory values are interpreted beneath ``data_root``; values
        already under the root are preserved; paths that escape the root raise a
        ``ValueError`` before the application starts.
    """
    # Normalize the configured location relative to the trusted data root.
    normalized_path = _normalize_local_path(configured_path, base_path=data_root)

    # Ensure the normalized directory cannot escape the configured data root.
    validated_path = _assert_within_directory(
        normalized_path,
        parent_directory=data_root,
        field_name=field_name,
    )

    # Return the safe absolute directory path for storage-related settings.
    return validated_path


def ensure_directory(path: Path) -> Path:
    """Create a directory and return the resulting absolute path.

    Why this function exists:
        Many runtime components need to lazily materialize local storage
        directories. Keeping that logic here avoids duplicating mkdir calls and
        makes directory creation semantics explicit.

    Parameters:
        path: Filesystem path that should exist as a directory.

    Returns:
        The absolute directory path after creation succeeds.

    Edge cases handled:
        Existing directories are accepted without error through
        ``exist_ok=True``; missing parent directories are created
        automatically; the function does not require the caller to normalize the
        path first.
    """
    # Create the entire directory tree so callers can immediately write files
    # beneath the returned location.
    path.mkdir(parents=True, exist_ok=True)

    # Return an absolute path to reduce ambiguity for downstream consumers.
    ensured_path = path.resolve(strict=False)

    # Return the created directory path to the caller.
    return ensured_path


def resolve_managed_data_path(base_directory: Path, relative_path: str) -> Path:
    """Resolve a file path safely beneath a managed base directory.

    Why this function exists:
        Storage code will eventually construct paths for raw uploads, parsed
        artifacts, and indexes. This helper prevents traversal outside the
        managed directory while still allowing nested relative paths.

    Parameters:
        base_directory: Trusted absolute directory under application control.
        relative_path: User- or pipeline-supplied relative path fragment.

    Returns:
        An absolute ``Path`` pointing to a location under ``base_directory``.

    Edge cases handled:
        The helper rejects absolute input paths, collapses redundant path
        segments, and raises ``ValueError`` if the normalized result escapes the
        base directory.
    """
    # Reject absolute paths because callers should only provide logical
    # filenames or nested relative paths inside the managed base directory.
    candidate_relative_path = Path(relative_path)
    if candidate_relative_path.is_absolute():
        raise ValueError("relative_path must be relative, not absolute")

    # Resolve the candidate path under the trusted base directory without
    # requiring the target file to exist yet.
    resolved_path = (base_directory / candidate_relative_path).resolve(strict=False)

    # Ensure the final path remains within the managed base directory after all
    # path normalization has been applied.
    _assert_within_directory(
        resolved_path,
        parent_directory=base_directory.resolve(strict=False),
        field_name="relative_path",
    )

    # Return the safe absolute path to the caller.
    return resolved_path


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    app_name: str = Field(default="Local AgentOps", alias="APP_NAME", min_length=3, max_length=100)
    environment: Literal["development", "test", "production"] = Field(
        default="development",
        alias="APP_ENV",
    )
    host: str = Field(default="127.0.0.1", alias="APP_HOST", min_length=1, max_length=255)
    port: int = Field(default=8000, alias="APP_PORT", ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="APP_LOG_LEVEL",
    )
    debug: bool = Field(default=True, alias="APP_DEBUG")
    data_root: Path = Field(default=Path("data"), alias="DATA_ROOT")
    raw_data_dir: Path = Field(default=Path("raw"), alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=Path("processed"), alias="PROCESSED_DATA_DIR")
    index_data_dir: Path = Field(default=Path("indexes"), alias="INDEX_DATA_DIR")
    eval_data_dir: Path = Field(default=Path("eval"), alias="EVAL_DATA_DIR")
    local_llm_provider: Literal["ollama", "llamacpp", "vllm", "custom"] = Field(
        default="ollama",
        alias="LOCAL_LLM_PROVIDER",
    )
    local_llm_model: str = Field(
        default="llama3.1",
        alias="LOCAL_LLM_MODEL",
        min_length=1,
        max_length=200,
    )
    local_llm_base_url: str = Field(
        default="http://127.0.0.1:11434",
        alias="LOCAL_LLM_BASE_URL",
        min_length=1,
        max_length=500,
    )
    local_llm_timeout_seconds: int = Field(
        default=60,
        alias="LOCAL_LLM_TIMEOUT_SECONDS",
        ge=1,
        le=600,
    )
    local_embedding_model: str = Field(
        default="bge-small-en-v1.5",
        alias="LOCAL_EMBEDDING_MODEL",
        min_length=1,
        max_length=200,
    )
    default_top_k: int = Field(default=5, alias="DEFAULT_TOP_K", ge=1, le=100)
    default_chunk_size: int = Field(default=800, alias="DEFAULT_CHUNK_SIZE", ge=100, le=8000)
    default_chunk_overlap: int = Field(default=120, alias="DEFAULT_CHUNK_OVERLAP", ge=0, le=4000)
    max_context_chunks: int = Field(default=12, alias="MAX_CONTEXT_CHUNKS", ge=1, le=100)
    citation_character_limit: int = Field(default=280, alias="CITATION_CHARACTER_LIMIT", ge=40, le=2000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    @field_validator(
        "app_name",
        "host",
        "local_llm_model",
        "local_llm_base_url",
        "local_embedding_model",
        mode="before",
    )
    @classmethod
    def strip_text_fields(cls, value: object) -> object:
        """Trim whitespace from text settings before validation.

        Why this function exists:
            Environment variables often gain accidental spaces when copied
            between shells or deployment files. Normalizing them early reduces
            avoidable validation failures and makes configuration more forgiving.

        Parameters:
            value: Raw value supplied by Pydantic before field coercion.

        Returns:
            The same value with leading and trailing whitespace removed when the
            input is a string.

        Edge cases handled:
            Non-string values are returned unchanged so Pydantic can perform its
            normal coercion and type-specific validation.
        """
        # Remove surrounding whitespace from string values while leaving all
        # other types untouched for standard Pydantic handling.
        if isinstance(value, str):
            return value.strip()

        # Return non-string values unchanged because they may already be typed.
        return value

    @field_validator("data_root", mode="before")
    @classmethod
    def normalize_data_root(cls, value: Path | str) -> Path:
        """Convert the configured data root into an absolute repository path.

        Why this function exists:
            The data root is the anchor for raw data, processed artifacts,
            indexes, and evaluation output. Converting it into an absolute path
            once prevents every downstream module from repeating that work.

        Parameters:
            value: Data root supplied through defaults or environment variables.

        Returns:
            An absolute normalized ``Path`` beneath the repository root.

        Edge cases handled:
            Relative roots are anchored to the repo root; absolute paths are
            preserved; nonexistent directories are allowed because creation can
            happen later during startup.
        """
        # Normalize the configured root relative to the repository root.
        normalized_root = _normalize_local_path(value, base_path=get_repo_root())

        # Return the normalized absolute root for later field validation.
        return normalized_root

    @model_validator(mode="after")
    def validate_directory_settings(self) -> "Settings":
        """Normalize and validate the managed data subdirectories.

        Why this function exists:
            The settings object contains several directory fields that must stay
            under ``data_root``. Validating them together ensures cross-field
            rules are enforced before the application starts.

        Parameters:
            This method operates on ``self`` after individual fields have been
            parsed, so no additional parameters are required.

        Returns:
            The same ``Settings`` instance with normalized absolute directory
            paths assigned back onto it.

        Edge cases handled:
            Relative subdirectory values are supported; absolute values are only
            accepted when they still live under ``data_root``; an overlap larger
            than the chunk size is rejected with a clear error.
        """
        # Normalize each managed subdirectory beneath the trusted data root so
        # every runtime component receives a safe absolute path.
        self.raw_data_dir = resolve_data_subdirectory(self.data_root, self.raw_data_dir, "RAW_DATA_DIR")
        self.processed_data_dir = resolve_data_subdirectory(
            self.data_root,
            self.processed_data_dir,
            "PROCESSED_DATA_DIR",
        )
        self.index_data_dir = resolve_data_subdirectory(
            self.data_root,
            self.index_data_dir,
            "INDEX_DATA_DIR",
        )
        self.eval_data_dir = resolve_data_subdirectory(self.data_root, self.eval_data_dir, "EVAL_DATA_DIR")

        # Prevent impossible chunking defaults where the overlap fully consumes
        # or exceeds the chunk size.
        if self.default_chunk_overlap >= self.default_chunk_size:
            raise ValueError("DEFAULT_CHUNK_OVERLAP must be smaller than DEFAULT_CHUNK_SIZE")

        # Return the validated settings object so Pydantic can finalize it.
        return self

    def create_data_directories(self) -> dict[str, Path]:
        """Create and return the managed application data directories.

        Why this function exists:
            Startup code and offline scripts both need a convenient way to
            materialize the local storage layout expected by ingestion and
            indexing workflows.

        Parameters:
            This method uses the already-validated settings stored on ``self``.

        Returns:
            A mapping from logical directory name to the ensured absolute path.

        Edge cases handled:
            Existing directories are reused without error; missing parent
            directories are created automatically.
        """
        # Ensure the data root exists before creating its child directories.
        ensured_data_root = ensure_directory(self.data_root)

        # Materialize each managed directory and expose them via stable keys so
        # callers can use the returned mapping programmatically.
        directories = {
            "data_root": ensured_data_root,
            "raw": ensure_directory(self.raw_data_dir),
            "processed": ensure_directory(self.processed_data_dir),
            "indexes": ensure_directory(self.index_data_dir),
            "eval": ensure_directory(self.eval_data_dir),
        }

        # Return the created directory mapping to the caller.
        return directories

    def resolve_raw_data_path(self, relative_path: str) -> Path:
        """Resolve a safe path under the raw-data directory.

        Why this function exists:
            Ingestion code will store uploaded documents and ticket exports
            beneath ``raw_data_dir``. A dedicated helper keeps those writes
            confined to the correct managed directory.

        Parameters:
            relative_path: Relative file path or nested fragment under
                ``raw_data_dir``.

        Returns:
            An absolute path beneath ``raw_data_dir``.

        Edge cases handled:
            Absolute paths and traversal attempts are rejected by the shared
            path resolver.
        """
        # Delegate the actual validation logic to the shared managed-path
        # resolver to avoid duplicating filesystem safety checks.
        return resolve_managed_data_path(self.raw_data_dir, relative_path)

    def resolve_processed_data_path(self, relative_path: str) -> Path:
        """Resolve a safe path under the processed-data directory.

        Why this function exists:
            Parsing, chunking, and normalization stages need a canonical place
            to store structured artifacts, and this helper documents that
            contract in the settings layer.

        Parameters:
            relative_path: Relative file path or nested fragment under
                ``processed_data_dir``.

        Returns:
            An absolute path beneath ``processed_data_dir``.

        Edge cases handled:
            The shared path resolver rejects absolute or escaping paths before
            any file operations occur.
        """
        # Use the shared safe resolver so processed artifacts stay under the
        # configured managed directory.
        return resolve_managed_data_path(self.processed_data_dir, relative_path)

    def resolve_index_data_path(self, relative_path: str) -> Path:
        """Resolve a safe path under the index-data directory.

        Why this function exists:
            Hybrid retrieval will persist lexical indexes, vector indexes, and
            metadata under a dedicated directory. This helper keeps those paths
            predictable and safe.

        Parameters:
            relative_path: Relative file path or nested fragment under
                ``index_data_dir``.

        Returns:
            An absolute path beneath ``index_data_dir``.

        Edge cases handled:
            Invalid absolute or traversal paths are rejected by the shared
            resolver before storage code can use them.
        """
        # Resolve the supplied index path beneath the managed index directory.
        return resolve_managed_data_path(self.index_data_dir, relative_path)

    def resolve_eval_data_path(self, relative_path: str) -> Path:
        """Resolve a safe path under the evaluation-data directory.

        Why this function exists:
            Evaluation fixtures and score reports should live under a separate
            managed directory so they do not mix with ingestion or retrieval
            artifacts.

        Parameters:
            relative_path: Relative file path or nested fragment under
                ``eval_data_dir``.

        Returns:
            An absolute path beneath ``eval_data_dir``.

        Edge cases handled:
            The shared path resolver blocks absolute input and directory
            traversal attempts.
        """
        # Resolve the supplied evaluation path beneath the managed eval
        # directory using the same safety checks as other storage helpers.
        return resolve_managed_data_path(self.eval_data_dir, relative_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance for the current process.

    Why this function exists:
        Configuration should be parsed once and shared across request handlers,
        background jobs, and service objects. Caching also makes repeated access
        inexpensive during import-heavy startup.

    Parameters:
        This helper does not accept parameters because the application uses a
        single environment-backed settings object per process.

    Returns:
        A validated ``Settings`` instance loaded from environment variables and
        defaults.

    Edge cases handled:
        Validation failures surface immediately during the first call, which is
        desirable because misconfiguration should stop the app before runtime
        side effects occur.
    """
    # Instantiate the settings object once so all callers share the same
    # validated configuration snapshot.
    settings = Settings()

    # Return the cached settings instance to the caller.
    return settings
