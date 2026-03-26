"""Tests for OutputController — LLM key-value → DCP positional array."""

import json
import pytest
from dcp_rag.core.schema import DcpSchema, FieldType
from dcp_rag.core.controller import OutputController, PlacementResult


# Custom schema for testing
KNOWLEDGE_SCHEMA = DcpSchema(
    id="knowledge:v1",
    description="Knowledge action schema",
    fields=("action", "domain", "detail", "confidence"),
    field_count=4,
    types={
        "action": FieldType(type="string", enum=["add", "replace", "flag", "remove"]),
        "domain": FieldType(type="string"),
        "detail": FieldType(type="string"),
        "confidence": FieldType(type="number", min=0, max=1),
    },
)


class TestOutputController:

    def test_basic_placement(self):
        """Key-value dict → correct positional order."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA)
        result = ctrl.place({
            "action": "replace",
            "domain": "auth",
            "detail": "jwt migration",
            "confidence": 0.9,
        })

        assert result.row == ["replace", "auth", "jwt migration", 0.9]
        assert result.is_valid
        assert result.schema_id == "knowledge:v1"

    def test_order_independent(self):
        """Input dict order should not affect output position."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA)
        result = ctrl.place({
            "confidence": 0.5,
            "detail": "timeout fix",
            "action": "flag",
            "domain": "payment",
        })

        assert result.row == ["flag", "payment", "timeout fix", 0.5]

    def test_missing_fields_become_none(self):
        """Missing keys produce None at the correct positions."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA)
        result = ctrl.place({"action": "flag", "domain": "payment"})

        assert result.row == ["flag", "payment", None, None]
        assert any("missing fields" in w for w in result.warnings)

    def test_extra_keys_ignored(self):
        """Unknown keys are silently dropped with a warning."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA)
        result = ctrl.place({
            "action": "add",
            "domain": "auth",
            "detail": "new feature",
            "confidence": 0.8,
            "extra_field": "should be ignored",
            "another": 42,
        })

        assert result.row == ["add", "auth", "new feature", 0.8]
        assert len(result.row) == 4  # no extra values
        assert any("ignored unknown keys" in w for w in result.warnings)

    def test_enum_validation(self):
        """Invalid enum value should produce validation warning."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA)
        result = ctrl.place({
            "action": "invalid_action",
            "domain": "auth",
            "detail": "test",
            "confidence": 0.5,
        })

        assert not result.is_valid
        assert any("enum" in w for w in result.warnings)

    def test_numeric_range_validation(self):
        """Out-of-range number should produce validation warning."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA)
        result = ctrl.place({
            "action": "add",
            "domain": "auth",
            "detail": "test",
            "confidence": 1.5,  # > max 1
        })

        assert not result.is_valid

    def test_strict_mode_raises(self):
        """Strict mode should raise ValueError on validation failure."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA, strict=True)

        with pytest.raises(ValueError, match="validation failed"):
            ctrl.place({
                "action": "bad_value",
                "domain": "auth",
                "detail": "test",
                "confidence": 0.5,
            })

    def test_strict_mode_valid(self):
        """Strict mode should not raise on valid input."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA, strict=True)
        result = ctrl.place({
            "action": "add",
            "domain": "auth",
            "detail": "test",
            "confidence": 0.5,
        })
        assert result.is_valid

    def test_to_json(self):
        """PlacementResult.to_json() should produce valid JSON array."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA)
        result = ctrl.place({
            "action": "add",
            "domain": "auth",
            "detail": "test",
            "confidence": 0.8,
        })

        j = result.to_json()
        parsed = json.loads(j)
        assert parsed == ["add", "auth", "test", 0.8]

    def test_place_batch(self):
        """place_batch should process multiple items."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA)
        items = [
            {"action": "add", "domain": "auth", "detail": "a", "confidence": 0.9},
            {"action": "flag", "domain": "db", "detail": "b", "confidence": 0.3},
        ]

        results = ctrl.place_batch(items)
        assert len(results) == 2
        assert results[0].row == ["add", "auth", "a", 0.9]
        assert results[1].row == ["flag", "db", "b", 0.3]

    def test_empty_dict(self):
        """Empty dict should produce all-None row."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA)
        result = ctrl.place({})

        assert result.row == [None, None, None, None]

    def test_properties(self):
        """Controller should expose schema and fields."""
        ctrl = OutputController(schema=KNOWLEDGE_SCHEMA)
        assert ctrl.schema.id == "knowledge:v1"
        assert ctrl.fields == ("action", "domain", "detail", "confidence")


class TestAutoBindMapping:
    """Tests for FieldMapping.auto_bind."""

    def test_flat_auto_bind(self):
        from dcp_rag.core.mapping import FieldMapping

        sample = {"level": "error", "service": "auth", "timestamp": 1234, "code": "E_TIMEOUT"}
        mapping = FieldMapping.auto_bind(
            schema_id="log:v1",
            fields=["level", "service", "timestamp", "code"],
            sample=sample,
        )

        assert mapping.paths["level"] == "level"
        assert mapping.paths["service"] == "service"
        resolved = mapping.resolve(sample)
        assert resolved["level"] == "error"

    def test_nested_auto_bind(self):
        from dcp_rag.core.mapping import FieldMapping

        sample = {
            "score": 0.9,
            "metadata": {"source": "docs/auth.md", "page": 12, "section": "JWT"},
        }
        mapping = FieldMapping.auto_bind(
            schema_id="test:v1",
            fields=["source", "page", "section", "score"],
            sample=sample,
        )

        assert mapping.paths["score"] == "score"
        assert mapping.paths["source"] == "metadata.source"
        assert mapping.paths["page"] == "metadata.page"

        resolved = mapping.resolve(sample)
        assert resolved["source"] == "docs/auth.md"
        assert resolved["score"] == 0.9

    def test_auto_bind_with_overrides(self):
        from dcp_rag.core.mapping import FieldMapping

        sample = {"score": 0.9, "metadata": {"file_path": "docs/auth.md"}}
        mapping = FieldMapping.auto_bind(
            schema_id="test:v1",
            fields=["source", "score"],
            sample=sample,
            overrides={"source": "metadata.file_path"},
        )

        assert mapping.paths["source"] == "metadata.file_path"
        resolved = mapping.resolve(sample)
        assert resolved["source"] == "docs/auth.md"

    def test_auto_bind_missing_field(self):
        from dcp_rag.core.mapping import FieldMapping

        sample = {"score": 0.9}
        mapping = FieldMapping.auto_bind(
            schema_id="test:v1",
            fields=["source", "score"],
            sample=sample,
        )

        # source not found, should not be in paths
        assert "source" not in mapping.paths
        resolved = mapping.resolve(sample)
        assert resolved.get("source") is None
        assert resolved["score"] == 0.9