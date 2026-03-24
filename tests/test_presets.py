"""Tests for Layer 2: DB presets."""

import pytest

from dcp_rag.core.presets import get_preset, list_presets
from dcp_rag.core.mapping import FieldMapping


class TestPresets:
    def test_list_presets(self):
        presets = list_presets()
        assert "pinecone" in presets
        assert "qdrant" in presets
        assert "weaviate" in presets
        assert "chroma" in presets
        assert "milvus" in presets

    def test_all_presets_return_field_mapping(self):
        for db_name in list_presets():
            mapping = get_preset(db_name)
            assert isinstance(mapping, FieldMapping)
            assert mapping.schema_id == "rag-chunk-meta:v1"

    def test_all_presets_have_required_fields(self):
        required = {"source", "page", "section", "score", "chunk_index"}
        for db_name in list_presets():
            mapping = get_preset(db_name)
            assert set(mapping.paths.keys()) == required, f"{db_name} missing fields"

    def test_unknown_preset(self):
        with pytest.raises(KeyError, match="no preset"):
            get_preset("unknown_db")

    def test_unknown_schema(self):
        with pytest.raises(KeyError):
            get_preset("pinecone", schema_id="nonexistent:v1")


class TestPresetResolution:
    """Verify each preset resolves correctly against its DB's response format."""

    def test_pinecone(self):
        mapping = get_preset("pinecone")
        source = {
            "id": "vec_123", "score": 0.92, "values": [],
            "metadata": {"source": "docs/auth.md", "page": 12, "section": "JWT", "chunk_index": 3},
        }
        result = mapping.resolve(source)
        assert result["source"] == "docs/auth.md"
        assert result["score"] == 0.92
        assert result["page"] == 12

    def test_qdrant(self):
        mapping = get_preset("qdrant")
        source = {
            "id": "abc", "score": 0.85,
            "payload": {"source": "docs/deploy.md", "page": None, "section": "Docker", "chunk_index": 1},
        }
        result = mapping.resolve(source)
        assert result["source"] == "docs/deploy.md"
        assert result["score"] == 0.85

    def test_weaviate(self):
        mapping = get_preset("weaviate")
        source = {
            "_additional": {"score": 0.76, "distance": 0.24},
            "properties": {"source": "readme.md", "page": 1, "section": "Intro", "chunk_index": 0},
        }
        result = mapping.resolve(source)
        assert result["score"] == 0.76
        assert result["source"] == "readme.md"

    def test_chroma(self):
        mapping = get_preset("chroma")
        source = {
            "id": "chr_1", "distance": 0.15,
            "metadata": {"source": "notes.md", "page": None, "section": "Summary", "chunk_index": 2},
        }
        result = mapping.resolve(source)
        assert result["score"] == 0.15
        assert result["source"] == "notes.md"

    def test_milvus(self):
        mapping = get_preset("milvus")
        source = {
            "id": 42, "distance": 0.33,
            "entity": {"source": "data.csv", "page": 5, "section": "Row 100", "chunk_index": 7},
        }
        result = mapping.resolve(source)
        assert result["score"] == 0.33
        assert result["source"] == "data.csv"
        assert result["page"] == 5
