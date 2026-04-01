"""Tests for Layer 1: FieldMapping."""

import pytest

from dcp_py.core.mapping import FieldMapping, resolve_path


class TestResolvePath:
    def test_flat(self):
        assert resolve_path({"score": 0.9}, "score") == 0.9

    def test_nested(self):
        assert resolve_path({"a": {"b": 1}}, "a.b") == 1

    def test_deep(self):
        assert resolve_path({"a": {"b": {"c": "x"}}}, "a.b.c") == "x"

    def test_missing(self):
        assert resolve_path({"a": 1}, "b") is None

    def test_missing_nested(self):
        assert resolve_path({"a": {"b": 1}}, "a.c") is None

    def test_none_intermediate(self):
        assert resolve_path({"a": None}, "a.b") is None

    def test_non_dict(self):
        assert resolve_path({"a": 42}, "a.b") is None


class TestFieldMapping:
    @pytest.fixture
    def pinecone_mapping(self):
        return FieldMapping(
            schema_id="rag-chunk-meta:v1",
            paths={
                "source": "metadata.source",
                "page": "metadata.page",
                "section": "metadata.section",
                "score": "score",
                "chunk_index": "metadata.chunk_index",
            },
        )

    def test_resolve_full(self, pinecone_mapping):
        source = {
            "score": 0.92,
            "metadata": {
                "source": "docs/auth.md",
                "page": None,
                "section": "JWT Config",
                "chunk_index": 3,
            },
        }
        result = pinecone_mapping.resolve(source)
        assert result == {
            "source": "docs/auth.md",
            "page": None,
            "section": "JWT Config",
            "score": 0.92,
            "chunk_index": 3,
        }

    def test_resolve_missing_fields(self, pinecone_mapping):
        source = {"score": 0.5, "metadata": {"source": "test.md"}}
        result = pinecone_mapping.resolve(source)
        assert result["source"] == "test.md"
        assert result["score"] == 0.5
        assert result["page"] is None
        assert result["section"] is None
        assert result["chunk_index"] is None

    def test_resolve_to_row(self, pinecone_mapping):
        source = {
            "score": 0.88,
            "metadata": {"source": "api.yaml", "page": 5, "section": "login", "chunk_index": 2},
        }
        fields = ("source", "page", "section", "score", "chunk_index")
        row = pinecone_mapping.resolve_to_row(source, fields)
        assert row == ["api.yaml", 5, "login", 0.88, 2]

    def test_with_overrides(self, pinecone_mapping):
        overridden = pinecone_mapping.with_overrides({"section": "metadata.heading"})
        assert overridden.paths["section"] == "metadata.heading"
        assert overridden.paths["source"] == "metadata.source"  # unchanged

        source = {"score": 0.7, "metadata": {"source": "x.md", "heading": "Setup"}}
        result = overridden.resolve(source)
        assert result["section"] == "Setup"

    def test_qdrant_mapping(self):
        mapping = FieldMapping(
            schema_id="rag-chunk-meta:v1",
            paths={
                "source": "payload.source",
                "page": "payload.page",
                "section": "payload.section",
                "score": "score",
                "chunk_index": "payload.chunk_index",
            },
        )
        source = {
            "id": "abc",
            "score": 0.85,
            "payload": {"source": "docs/deploy.md", "page": None, "section": "Docker", "chunk_index": 1},
        }
        result = mapping.resolve(source)
        assert result["source"] == "docs/deploy.md"
        assert result["score"] == 0.85

    def test_weaviate_mapping(self):
        mapping = FieldMapping(
            schema_id="rag-chunk-meta:v1",
            paths={
                "source": "properties.source",
                "page": "properties.page",
                "section": "properties.section",
                "score": "_additional.score",
                "chunk_index": "properties.chunk_index",
            },
        )
        source = {
            "_additional": {"score": 0.76, "distance": 0.24},
            "properties": {"source": "readme.md", "page": 1, "section": "Intro", "chunk_index": 0},
        }
        result = mapping.resolve(source)
        assert result["score"] == 0.76
        assert result["source"] == "readme.md"
