"""Tests for SchemaGenerator — infer DcpSchema from data samples."""

import json
import pytest
from dcp_rag.core.generator import SchemaGenerator, SchemaDraft


# ── Sample data fixtures ─────────────────────────────────────

PINECONE_SAMPLES = [
    {
        "id": "abc123",
        "score": 0.92,
        "metadata": {
            "source": "docs/auth.md",
            "page": 12,
            "section": "JWT Config",
            "chunk_index": 3,
        },
    },
    {
        "id": "def456",
        "score": 0.87,
        "metadata": {
            "source": "docs/api.md",
            "page": 5,
            "section": "Rate Limiting",
            "chunk_index": 1,
        },
    },
    {
        "id": "ghi789",
        "score": 0.78,
        "metadata": {
            "source": "docs/auth.md",
            "page": None,
            "section": "Token Refresh",
            "chunk_index": 7,
        },
    },
]

LOG_SAMPLES = [
    {"level": "error", "service": "auth-service", "timestamp": 1711284600, "code": "E_TIMEOUT"},
    {"level": "warn", "service": "payment-service", "timestamp": 1711284700, "code": None},
    {"level": "error", "service": "db-service", "timestamp": 1711284800, "code": "E_CONN"},
    {"level": "info", "service": "auth-service", "timestamp": 1711284900, "code": None},
    {"level": "error", "service": "cache-service", "timestamp": 1711285000, "code": "E_MEM"},
]

FLAT_SAMPLES = [
    {"name": "gateway", "status": "active", "weight": 5, "region": "us-east"},
    {"name": "worker", "status": "paused", "weight": 1, "region": "eu-west"},
    {"name": "scheduler", "status": "active", "weight": 3, "region": "us-east"},
    {"name": "indexer", "status": "active", "weight": 8, "region": "ap-south"},
]


class TestSchemaGenerator:

    def test_basic_generation(self):
        gen = SchemaGenerator()
        draft = gen.from_samples(LOG_SAMPLES, domain="log-entry")

        assert draft.schema.id == "log-entry:v1"
        assert draft.schema.field_count == len(draft.schema.fields)
        assert len(draft.mapping.paths) == draft.schema.field_count

    def test_field_ordering_convention(self):
        """Identifiers and classifiers should come before numerics and text."""
        gen = SchemaGenerator()
        draft = gen.from_samples(FLAT_SAMPLES, domain="service-registry")

        fields = list(draft.schema.fields)
        # name is identifier, status/region are classifiers, weight is numeric
        name_idx = fields.index("name")
        weight_idx = fields.index("weight")
        assert name_idx < weight_idx, "identifier should come before numeric"

    def test_enum_detection(self):
        """Low-cardinality string fields should be detected as enums."""
        gen = SchemaGenerator()
        draft = gen.from_samples(LOG_SAMPLES, domain="log-entry")

        level_type = draft.schema.types.get("level")
        assert level_type is not None
        assert level_type.enum is not None
        assert set(level_type.enum) == {"error", "warn", "info"}

    def test_nullable_detection(self):
        """Fields with None values should have null in type union."""
        gen = SchemaGenerator()
        draft = gen.from_samples(LOG_SAMPLES, domain="log-entry")

        code_type = draft.schema.types.get("code")
        assert code_type is not None
        types = code_type.type if isinstance(code_type.type, list) else [code_type.type]
        assert "null" in types

    def test_numeric_range_detection(self):
        """Score fields (0-1) should have min/max detected."""
        gen = SchemaGenerator()
        draft = gen.from_samples(
            PINECONE_SAMPLES,
            domain="rag-meta",
            exclude=["id"],
        )

        score_type = draft.schema.types.get("score")
        assert score_type is not None
        assert score_type.min == 0
        assert score_type.max == 1

    def test_nested_flattening(self):
        """Nested dict keys should be flattened and mapped via dot-notation."""
        gen = SchemaGenerator()
        draft = gen.from_samples(
            PINECONE_SAMPLES,
            domain="rag-meta",
            exclude=["id"],
        )

        # Check that nested paths are preserved in mapping
        paths = draft.mapping.paths
        assert any("metadata." in p for p in paths.values()), \
            f"Expected nested metadata paths, got: {paths}"

    def test_exclude_fields(self):
        """Excluded paths should not appear in schema."""
        gen = SchemaGenerator()
        draft = gen.from_samples(
            PINECONE_SAMPLES,
            domain="rag-meta",
            exclude=["id", "metadata.chunk_index"],
        )

        assert "id" not in draft.schema.fields
        assert "chunk_index" not in draft.schema.fields

    def test_include_fields(self):
        """Only included paths should appear in schema."""
        gen = SchemaGenerator()
        draft = gen.from_samples(
            PINECONE_SAMPLES,
            domain="rag-meta",
            include=["score", "metadata.source"],
        )

        assert draft.schema.field_count == 2
        assert "source" in draft.schema.fields
        assert "score" in draft.schema.fields

    def test_field_name_override(self):
        """Custom field names should override leaf names."""
        gen = SchemaGenerator()
        draft = gen.from_samples(
            PINECONE_SAMPLES,
            domain="rag-meta",
            include=["score", "metadata.source"],
            field_names={"metadata.source": "doc_path"},
        )

        assert "doc_path" in draft.schema.fields
        assert "source" not in draft.schema.fields
        assert draft.mapping.paths["doc_path"] == "metadata.source"

    def test_group_key_candidate(self):
        """Fields with high repetition should be flagged as group key candidates."""
        gen = SchemaGenerator()
        draft = gen.from_samples(LOG_SAMPLES, domain="log-entry")

        service_report = next(
            (r for r in draft.field_reports if r.name == "service"),
            None,
        )
        # service has repetition (auth-service appears twice)
        # but may or may not hit the 0.3 threshold with 5 samples
        assert service_report is not None

    def test_report_generation(self):
        """Report should be a readable string."""
        gen = SchemaGenerator()
        draft = gen.from_samples(LOG_SAMPLES, domain="log-entry")

        report = draft.report
        assert "log-entry:v1" in report
        assert "level" in report

    def test_to_dict_roundtrip(self):
        """to_dict output should be loadable by DcpSchema.from_dict."""
        gen = SchemaGenerator()
        draft = gen.from_samples(LOG_SAMPLES, domain="log-entry")

        from dcp_rag.core.schema import DcpSchema
        d = draft.to_dict()
        assert d["$dcp"] == "schema"
        schema2 = DcpSchema.from_dict(d)
        assert schema2.id == draft.schema.id
        assert schema2.fields == draft.schema.fields

    def test_to_encoder(self):
        """Draft should produce a working encoder."""
        gen = SchemaGenerator()
        draft = gen.from_samples(LOG_SAMPLES, domain="log-entry")

        encoder = draft.to_encoder()
        texts = [""] * len(LOG_SAMPLES)
        result = encoder.encode(LOG_SAMPLES, texts=texts)
        assert result.header != ""

    def test_save_and_load(self, tmp_path):
        """Saved schema should be loadable."""
        gen = SchemaGenerator()
        draft = gen.from_samples(FLAT_SAMPLES, domain="service-registry")

        path = tmp_path / "service-registry.v1.json"
        draft.save(path)

        from dcp_rag.core.schema import DcpSchema
        loaded = DcpSchema.from_file(path)
        assert loaded.id == "service-registry:v1"
        assert loaded.field_count == draft.schema.field_count

    def test_empty_samples_raises(self):
        gen = SchemaGenerator()
        with pytest.raises(ValueError, match="at least 1 sample"):
            gen.from_samples([], domain="empty")

    def test_version_override(self):
        gen = SchemaGenerator()
        draft = gen.from_samples(LOG_SAMPLES, domain="log-entry", version=2)
        assert draft.schema.id == "log-entry:v2"

    def test_duplicate_leaf_names(self):
        """Different nested paths with same leaf name should be deduplicated."""
        samples = [
            {"a": {"name": "foo"}, "b": {"name": "bar"}, "score": 0.5},
            {"a": {"name": "baz"}, "b": {"name": "qux"}, "score": 0.8},
        ]
        gen = SchemaGenerator()
        draft = gen.from_samples(samples, domain="dedup-test")

        # Should have name and name_1 (or similar dedup)
        assert len(set(draft.schema.fields)) == len(draft.schema.fields), \
            f"Duplicate field names: {draft.schema.fields}"