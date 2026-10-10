"""Configuration.

Plain dataclasses so there's no hard runtime dependency; :func:`load_config` is a
thin JSON adapter over them. Unknown keys raise rather than being ignored, so a
typo fails loudly instead of silently producing a different index than expected.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import ConfigurationError

__all__ = [
    "DEFAULT_EXCLUDED_DIRS",
    "DEFAULT_EXCLUDED_GLOBS",
    "SECRET_LIKE_FILENAMES",
    "LANGUAGE_BY_EXTENSION",
    "ChunkingConfig",
    "IngestionConfig",
    "RetrievalConfig",
    "EmbeddingConfig",
    "GenerationConfig",
    "OrbitConfig",
    "load_config",
    "config_to_dict",
]


DEFAULT_EXCLUDED_DIRS: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".bzr",
    ".tox",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "node_modules",
    "bower_components",
    ".next",
    ".nuxt",
    ".svelte-kit",
    "dist",
    "build",
    "out",
    "target",
    "bin",
    "obj",
    ".gradle",
    ".idea",
    ".vscode",
    ".vs",
    "vendor",
    "third_party",
    "Pods",
    "DerivedData",
    ".terraform",
    ".serverless",
    ".cache",
    "coverage",
    "htmlcov",
    ".orbit",
)

DEFAULT_EXCLUDED_GLOBS: tuple[str, ...] = (
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.o",
    "*.a",
    "*.lib",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.class",
    "*.jar",
    "*.war",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.gz",
    "*.bz2",
    "*.xz",
    "*.7z",
    "*.rar",
    "*.pdf",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
    "*.ico",
    "*.webp",
    "*.svg",
    "*.mp3",
    "*.mp4",
    "*.wav",
    "*.avi",
    "*.mov",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.otf",
    "*.eot",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.lock",
    "*.log",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.snap",
)

SECRET_LIKE_FILENAMES: tuple[str, ...] = (
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".envrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "_netrc",
    "credentials",
    ".htpasswd",
    "service-account.json",
)

SECRET_CONTENT_PATTERNS: tuple[str, ...] = (
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "-----BEGIN PGP PRIVATE KEY BLOCK-----",
    "AKIA[0-9A-Z]{16}",
)

LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".txt": "text",
    ".text": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".py": "python",
    ".pyi": "python",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".css": "css",
    ".scss": "scss",
    ".make": "make",
    ".mk": "make",
    ".dockerfile": "dockerfile",
    ".gradle": "gradle",
    ".proto": "protobuf",
}


@dataclass(frozen=True)
class ChunkingConfig:
    """Deterministic chunking parameters.

    The chunker is line-aware: it never splits a line, so ``start_line`` and
    ``end_line`` always map back to real lines in the source file.
    """

    max_tokens: int = 400
    overlap_tokens: int = 60
    max_chars: int = 0  # 0 = derive from max_tokens
    tokens_per_char: float = 4.0
    min_chunk_tokens: int = 20
    respect_blank_lines: bool = True
    respect_symbol_boundaries: bool = True

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ConfigurationError("chunking.max_tokens must be > 0")
        if self.overlap_tokens < 0:
            raise ConfigurationError("chunking.overlap_tokens must be >= 0")
        if self.overlap_tokens >= self.max_tokens:
            raise ConfigurationError(
                "chunking.overlap_tokens must be smaller than chunking.max_tokens"
            )
        if self.tokens_per_char <= 0:
            raise ConfigurationError("chunking.tokens_per_char must be > 0")
        if self.min_chunk_tokens < 0:
            raise ConfigurationError("chunking.min_chunk_tokens must be >= 0")

    @property
    def max_chars_effective(self) -> int:
        return self.max_chars if self.max_chars > 0 else int(self.max_tokens * self.tokens_per_char)

    @property
    def overlap_chars(self) -> int:
        return int(self.overlap_tokens * self.tokens_per_char)


@dataclass(frozen=True)
class IngestionConfig:
    """Which files are picked up, and how failures are handled."""

    extra_excluded_dirs: tuple[str, ...] = ()
    extra_excluded_globs: tuple[str, ...] = ()
    included_extensions: tuple[str, ...] = ()  # empty = all known extensions
    follow_symlinks: bool = False
    max_file_bytes: int = 2_000_000
    skip_secret_like_files: bool = True
    scan_content_for_secrets: bool = False
    secret_content_scan_bytes: int = 8192
    keep_going_on_error: bool = True
    encoding: str = "utf-8"
    decode_errors: str = "strict"
    text_sniff_bytes: int = 8192

    def __post_init__(self) -> None:
        if self.max_file_bytes < 0:
            raise ConfigurationError("ingestion.max_file_bytes must be >= 0")

    def effective_excluded_dirs(self) -> set[str]:
        return set(DEFAULT_EXCLUDED_DIRS) | set(self.extra_excluded_dirs)

    def effective_excluded_globs(self) -> set[str]:
        return set(DEFAULT_EXCLUDED_GLOBS) | set(self.extra_excluded_globs)

    def allowed_extensions(self) -> set[str] | None:
        return set(self.included_extensions) or None


@dataclass(frozen=True)
class RetrievalConfig:
    """Hybrid retrieval parameters.

    ``fusion`` is one of ``rrf`` (Cormack et al. SIGIR 2009; rank-based, so it
    doesn't care that BM25 and cosine live on different scales), ``weighted``
    (min-max normalised, weighted sum), or ``lexical_only``.
    """

    limit: int = 10
    candidate_multiplier: int = 4
    fusion: str = "rrf"
    lexical_weight: float = 1.0
    vector_weight: float = 1.0
    rrf_k: int = 60
    lexical_k1: float = 1.2
    lexical_b: float = 0.75
    min_lexical_score: float = 0.0
    # Unrelated English still shares character n-grams, so cosine never quite
    # hits zero and a 0.0 floor lets the whole corpus through. 0.05 cuts the tail.
    min_vector_score: float = 0.05
    min_normalized_score: float = 0.0
    dedupe: bool = True
    snippet_chars: int = 320
    vector_batch_size: int = 64
    metadata_filter_mode: str = "all"

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ConfigurationError("retrieval.limit must be > 0")
        if self.candidate_multiplier < 1:
            raise ConfigurationError("retrieval.candidate_multiplier must be >= 1")
        if self.fusion not in ("rrf", "weighted", "lexical_only"):
            raise ConfigurationError(
                "retrieval.fusion must be one of 'rrf', 'weighted', 'lexical_only'"
            )
        if self.lexical_weight < 0 or self.vector_weight < 0:
            raise ConfigurationError("retrieval weights must be >= 0")
        if self.rrf_k <= 0:
            raise ConfigurationError("retrieval.rrf_k must be > 0")
        if not 0.0 <= self.lexical_b <= 1.0:
            raise ConfigurationError("retrieval.lexical_b must be in [0, 1]")

    @property
    def lexical_only(self) -> bool:
        return self.fusion == "lexical_only"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding provider configuration.

    ``provider`` is ``hashing`` (built-in, offline, default), ``hf``
    (mean-pooled Transformers, needs the ``embeddings`` extra), ``openai_compatible``
    (any HTTP endpoint speaking that schema, only built when ``base_url`` is set),
    or ``none`` for lexical-only.
    """

    provider: str = "hashing"
    dimension: int = 512
    model: str = "BAAI/bge-small-en-v1.5"
    model_path: str = ""
    base_url: str = ""
    api_key_env: str = ""
    batch_size: int = 32
    device: str = "auto"
    normalize: bool = True
    lowercase: bool = True
    max_length: int = 512
    cache: bool = True

    def __post_init__(self) -> None:
        valid = ("hashing", "hf", "openai_compatible", "none")
        if self.provider not in valid:
            raise ConfigurationError(
                f"embedding.provider must be one of {valid}, got {self.provider!r}"
            )
        if self.provider == "hashing" and self.dimension <= 0:
            raise ConfigurationError("embedding.dimension must be > 0")

    @property
    def is_remote(self) -> bool:
        return self.provider == "openai_compatible" and bool(self.base_url)

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        # Read from the process environment only; never store keys in config.
        return os.environ.get(self.api_key_env) or None


@dataclass(frozen=True)
class GenerationConfig:
    """Optional answer generation.

    Disabled by default. Orbit is fully usable as a search and evidence engine
    without any generator.
    """

    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str = ""
    api_key_env: str = ""
    model: str = "local-model"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    system_prompt: str = (
        "You are Orbit's answer generator. You receive numbered EVIDENCE passages with "
        "source paths and content hashes. Rules: (1) Answer only from the evidence. "
        "(2) Label anything you infer as inference, not evidence. (3) If the evidence is "
        "insufficient, say so explicitly and answer nothing further. (4) Cite claims by "
        "the [n] number of the evidence passage you used. (5) Never invent a source path "
        "or line number that is not present in the evidence."
    )

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env) or None


@dataclass(frozen=True)
class OrbitConfig:
    """Top-level configuration object."""

    paths: tuple[str, ...] = ()
    db_path: str = ".orbit/orbit.db"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    context: ContextBudgetConfig = field(default_factory=lambda: ContextBudgetConfig())
    exclude_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.db_path:
            raise ConfigurationError("db_path must not be empty")
        if self.chunking is None or self.ingestion is None:
            raise ConfigurationError("sub-configs must be provided")

    def with_db_path(self, db_path: str) -> OrbitConfig:
        return replace(self, db_path=db_path)

    def with_paths(self, paths: Iterable[str]) -> OrbitConfig:
        return replace(self, paths=tuple(paths))

    def with_exclude_paths(self, paths: Iterable[str]) -> OrbitConfig:
        return replace(self, exclude_paths=tuple(paths))

    def with_embedding_provider(self, provider: str, **kwargs: Any) -> OrbitConfig:
        return replace(self, embedding=replace(self.embedding, provider=provider, **kwargs))


@dataclass(frozen=True)
class ContextBudgetConfig:
    """Parameters for the context assembler."""

    token_budget: int = 4000
    max_blocks: int = 12
    dedupe_hash: bool = True
    dedupe_similarity: bool = True
    similarity_threshold: float = 0.92
    min_score_ratio: float = 0.35
    include_metadata: bool = True
    ordering: str = "score"

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ConfigurationError("context.token_budget must be > 0")
        if self.max_blocks <= 0:
            raise ConfigurationError("context.max_blocks must be > 0")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ConfigurationError("context.similarity_threshold must be in [0, 1]")
        if not 0.0 <= self.min_score_ratio <= 1.0:
            raise ConfigurationError("context.min_score_ratio must be in [0, 1]")
        if self.ordering not in ("score", "document"):
            raise ConfigurationError("context.ordering must be 'score' or 'document'")


# ---
# Config loading
# ---
_SECTION_BUILDERS = {
    "chunking": ChunkingConfig,
    "ingestion": IngestionConfig,
    "retrieval": RetrievalConfig,
    "embedding": EmbeddingConfig,
    "generation": GenerationConfig,
    "context": ContextBudgetConfig,
}


def config_to_dict(config: OrbitConfig) -> dict[str, Any]:
    """Serialise a config to a plain dict (inverse of :func:`load_config`)."""
    out: dict[str, Any] = {
        "paths": list(config.paths),
        "db_path": config.db_path,
        "exclude_paths": list(config.exclude_paths),
    }
    for name, cls in _SECTION_BUILDERS.items():
        section: Any = getattr(config, name)
        out[name] = dict(section.__dict__)
    return out


def _coerce_section(name: str, cls: type, raw: Mapping[str, Any]) -> Any:
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"config section {name!r} must be a table/object")
    valid_fields = set(getattr(cls, "__dataclass_fields__", {}))
    unknown = set(raw) - valid_fields
    if unknown:
        raise ConfigurationError(
            f"unknown key(s) {sorted(unknown)} in config section {name!r}; "
            f"valid keys: {sorted(valid_fields)}"
        )
    try:
        return cls(**{k: _coerce_value(v) for k, v in raw.items()})
    except ConfigurationError:
        raise
    except TypeError as exc:
        raise ConfigurationError(f"invalid value in section {name!r}: {exc}") from exc


def _coerce_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return {k: _coerce_value(v) for k, v in value.items()}
    return value


def load_config(path: str | os.PathLike[str] | None = None) -> OrbitConfig:
    """Load an :class:`OrbitConfig` from a JSON file.

    Unknown keys raise :class:`ConfigurationError` rather than being silently
    ignored, so a typo in a config file fails loudly instead of producing an
    index the user does not expect.
    """
    if path is None:
        return OrbitConfig()
    p = Path(path)
    if not p.is_file():
        raise ConfigurationError(f"config file not found: {path}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"config file {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"config file {path} must contain a JSON object")

    kwargs: dict[str, Any] = {}
    for key in ("paths", "db_path", "exclude_paths"):
        if key in raw:
            val = _coerce_value(raw.pop(key))
            if key in ("paths", "exclude_paths") and not isinstance(val, tuple):
                raise ConfigurationError(f"{key!r} must be a list")
            kwargs[key] = val
    for name, cls in _SECTION_BUILDERS.items():
        if name in raw:
            section_raw = raw.pop(name)
            if not isinstance(section_raw, Mapping):
                raise ConfigurationError(f"config section {name!r} must be a table/object")
            kwargs[name] = _coerce_section(name, cls, section_raw)
    if raw:
        raise ConfigurationError(
            f"unknown top-level config key(s): {sorted(raw)}; "
            f"valid keys: {sorted(['paths', 'db_path', 'exclude_paths', *_SECTION_BUILDERS])}"
        )
    return OrbitConfig(**kwargs)
