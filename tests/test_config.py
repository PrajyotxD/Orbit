"""Configuration layer: defaults, validation, JSON load/serialise."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbit.config import (
    DEFAULT_EXCLUDED_DIRS,
    SECRET_LIKE_FILENAMES,
    ChunkingConfig,
    ContextBudgetConfig,
    EmbeddingConfig,
    GenerationConfig,
    IngestionConfig,
    OrbitConfig,
    RetrievalConfig,
    config_to_dict,
    load_config,
)
from orbit.errors import ConfigurationError, OrbitError


# --------------------------------------------------------------------- defaults
def test_default_config_is_usable_offline(tmp_path: Path) -> None:
    cfg = OrbitConfig()
    assert cfg.embedding.provider == "hashing"
    assert cfg.embedding.is_remote is False
    assert cfg.generation.enabled is False
    assert cfg.generation.base_url == ""
    assert cfg.retrieval.fusion == "rrf"
    assert cfg.retrieval.rrf_k == 60


def test_defaults_never_require_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert EmbeddingConfig().api_key() is None
    assert GenerationConfig().api_key() is None


def test_api_key_read_from_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_TEST_KEY", "s3cret")
    assert EmbeddingConfig(api_key_env="ORBIT_TEST_KEY").api_key() == "s3cret"
    assert GenerationConfig(api_key_env="ORBIT_TEST_KEY").api_key() == "s3cret"
    assert EmbeddingConfig(api_key_env="ORBIT_ABSENT_KEY").api_key() is None


def test_remote_embedder_requires_explicit_base_url() -> None:
    cfg = EmbeddingConfig(provider="openai_compatible", base_url="")
    assert cfg.is_remote is False
    assert EmbeddingConfig(provider="openai_compatible", base_url="http://x").is_remote is True


def test_chunking_derived_properties() -> None:
    cfg = ChunkingConfig(max_tokens=100, tokens_per_char=4.0)
    assert cfg.max_chars_effective == 400
    assert cfg.overlap_chars == 240
    explicit = ChunkingConfig(max_tokens=100, max_chars=1234)
    assert explicit.max_chars_effective == 1234


def test_effective_exclusions_include_defaults() -> None:
    ing = IngestionConfig(extra_excluded_dirs=("custom_dir",), extra_excluded_globs=("*.zzz",))
    dirs = ing.effective_excluded_dirs()
    globs = ing.effective_excluded_globs()
    assert "node_modules" in dirs and "custom_dir" in dirs
    assert "*.zip" in globs and "*.zzz" in globs
    assert all(d in DEFAULT_EXCLUDED_DIRS for d in ("node_modules", ".git", "dist"))
    assert ".env" in SECRET_LIKE_FILENAMES and "id_rsa" in SECRET_LIKE_FILENAMES


def test_allowed_extensions_none_when_empty() -> None:
    assert IngestionConfig().allowed_extensions() is None
    assert IngestionConfig(included_extensions=(".md",)).allowed_extensions() == {".md"}


def test_retrieval_lexical_only_property() -> None:
    assert RetrievalConfig(fusion="lexical_only").lexical_only is True
    assert RetrievalConfig(fusion="rrf").lexical_only is False


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (ChunkingConfig, {"max_tokens": 0}),
        (ChunkingConfig, {"max_tokens": -1}),
        (ChunkingConfig, {"max_tokens": 100, "overlap_tokens": 100}),
        (ChunkingConfig, {"max_tokens": 100, "overlap_tokens": 101}),
        (ChunkingConfig, {"overlap_tokens": -1}),
        (ChunkingConfig, {"tokens_per_char": 0}),
        (ChunkingConfig, {"min_chunk_tokens": -1}),
        (RetrievalConfig, {"limit": 0}),
        (RetrievalConfig, {"candidate_multiplier": 0}),
        (RetrievalConfig, {"fusion": "bogus"}),
        (RetrievalConfig, {"lexical_weight": -0.1}),
        (RetrievalConfig, {"vector_weight": -0.1}),
        (RetrievalConfig, {"rrf_k": 0}),
        (RetrievalConfig, {"lexical_b": 1.5}),
        (RetrievalConfig, {"lexical_b": -0.1}),
        (EmbeddingConfig, {"provider": "definitely-not-real"}),
        (EmbeddingConfig, {"provider": "hashing", "dimension": 0}),
        (EmbeddingConfig, {"provider": "hashing", "dimension": -4}),
        (IngestionConfig, {"max_file_bytes": -1}),
        (ContextBudgetConfig, {"token_budget": 0}),
        (ContextBudgetConfig, {"max_blocks": 0}),
        (ContextBudgetConfig, {"similarity_threshold": 1.5}),
        (ContextBudgetConfig, {"min_score_ratio": -0.2}),
        (ContextBudgetConfig, {"ordering": "random"}),
        (OrbitConfig, {"db_path": ""}),
    ],
)
def test_invalid_config_is_rejected_loudly(factory, kwargs) -> None:
    with pytest.raises(ConfigurationError):
        factory(**kwargs)


def test_configuration_error_is_machine_readable() -> None:
    err = ConfigurationError("boom", detail="extra")
    assert isinstance(err, OrbitError)
    assert err.code == "configuration_error"
    assert err.as_dict() == {"error": "configuration_error", "message": "boom", "detail": "extra"}
    assert str(err) == "boom (extra)"


def test_config_is_frozen() -> None:
    cfg = OrbitConfig()
    with pytest.raises(Exception):
        cfg.db_path = "other.db"  # type: ignore[misc]


# ----------------------------------------------------------------- derived cfgs
def test_with_helpers_return_new_objects(tmp_path: Path) -> None:
    cfg = OrbitConfig()
    other = cfg.with_db_path(str(tmp_path / "x.db"))
    assert other is not cfg
    assert cfg.db_path != other.db_path
    assert cfg.with_paths(["a", "b"]).paths == ("a", "b")
    assert cfg.with_embedding_provider("none").embedding.provider == "none"
    assert cfg.embedding.provider == "hashing"


# ------------------------------------------------------------------- json load
def test_load_config_roundtrip(tmp_path: Path) -> None:
    raw = {
        "paths": ["docs", "src"],
        "db_path": "kb/orbit.db",
        "exclude_paths": ["vendor"],
        "chunking": {"max_tokens": 256, "overlap_tokens": 32},
        "retrieval": {"fusion": "weighted", "limit": 3, "rrf_k": 7},
        "embedding": {"provider": "none"},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.paths == ("docs", "src")
    assert cfg.exclude_paths == ("vendor",)
    assert cfg.retrieval.fusion == "weighted"
    assert cfg.retrieval.rrf_k == 7
    assert cfg.embedding.provider == "none"
    assert cfg.chunking.max_tokens == 256


def test_load_config_none_returns_defaults() -> None:
    assert load_config(None) == OrbitConfig()


def test_config_to_dict_is_json_serialisable(tmp_path: Path) -> None:
    cfg = OrbitConfig(db_path=str(tmp_path / "d.db")).with_paths(["x"])
    d = config_to_dict(cfg)
    json.dumps(d)  # must not raise
    assert d["paths"] == ["x"]
    assert d["db_path"] == str(tmp_path / "d.db")
    assert set(d) >= {"chunking", "retrieval", "embedding", "context", "ingestion", "generation"}


def test_unknown_keys_are_not_silently_ignored(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"

    p.write_text(json.dumps({"retrieval": {"totally_bogus": 1}}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown key"):
        load_config(p)

    p.write_text(json.dumps({"totally_bogus": 1}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown top-level"):
        load_config(p)


def test_malformed_config_files_raise(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"

    p.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not valid JSON"):
        load_config(p)

    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must contain a JSON object"):
        load_config(p)

    p.write_text(json.dumps({"retrieval": [1, 2]}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be a table"):
        load_config(p)

    p.write_text(json.dumps({"paths": "notalist"}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be a list"):
        load_config(p)

    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "does-not-exist.json")


def test_wrong_value_type_inside_section_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"retrieval": {"rrf_k": "sixty"}}), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(p)


def test_derived_values_from_loaded_config(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(
        json.dumps({"chunking": {"max_tokens": 100, "overlap_tokens": 10, "tokens_per_char": 2.0}}),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.chunking.max_chars_effective == 200
    assert cfg.chunking.overlap_chars == 20