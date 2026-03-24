"""Tests for Layer 0: DcpSchema."""

import json
import pytest
from pathlib import Path

from dcp_rag.core.schema import DcpSchema, FieldType, SchemaRegistry, load_default_registry


SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


class TestFieldType:
    def test_string_valid(self):
        ft = FieldType(type="string")
        assert ft.validate("hello") is None

    def test_string_invalid(self):
        ft = FieldType(type="string")
        assert ft.validate(42) is not None

    def test_nullable(self):
        ft = FieldType(type=["string", "null"])
        assert ft.validate(None) is None
        assert ft.validate("ok") is None

    def test_not_nullable(self):
        ft = FieldType(type="string")
        assert ft.validate(None) is not None

    def test_number_range(self):
        ft = FieldType(type="number", min=0, max=1)
        assert ft.validate(0.5) is None
        assert ft.validate(-0.1) is not None
        assert ft.validate(1.1) is not None

    def test_enum(self):
        ft = FieldType(type="string", enum=["a", "b", "c"])
        assert ft.validate("a") is None
        assert ft.validate("z") is not None

    def test_boolean(self):
        ft = FieldType(type="boolean")
        assert ft.validate(True) is None
        assert ft.validate(1) is not None  # int is not bool


class TestDcpSchema:
    @pytest.fixture
    def chunk_meta(self):
        return DcpSchema.from_file(SCHEMAS_DIR / "rag-chunk-meta.v1.json")

    def test_load(self, chunk_meta):
        assert chunk_meta.id == "rag-chunk-meta:v1"
        assert chunk_meta.field_count == 5
        assert chunk_meta.fields == ("source", "page", "section", "score", "chunk_index")

    def test_full_mask(self, chunk_meta):
        assert chunk_meta.full_mask == 0b11111  # 31

    def test_field_bit(self, chunk_meta):
        assert chunk_meta.field_bit("source") == 0b10000  # 16
        assert chunk_meta.field_bit("score") == 0b00010   # 2

    def test_fields_from_mask(self, chunk_meta):
        assert chunk_meta.fields_from_mask(0b10010) == ("source", "score")
        assert chunk_meta.fields_from_mask(0b10110) == ("source", "section", "score")

    def test_cutdown_id_full(self, chunk_meta):
        assert chunk_meta.cutdown_id(0b11111) == "rag-chunk-meta:v1"

    def test_cutdown_id_partial(self, chunk_meta):
        assert chunk_meta.cutdown_id(0b10010) == "rag-chunk-meta:v1#12"

    def test_s_header_full(self, chunk_meta):
        h = chunk_meta.s_header()
        assert h == ["$S", "rag-chunk-meta:v1", 5, "source", "page", "section", "score", "chunk_index"]

    def test_s_header_cutdown(self, chunk_meta):
        h = chunk_meta.s_header(0b10010)
        assert h == ["$S", "rag-chunk-meta:v1#12", 2, "source", "score"]

    def test_validate_row_valid(self, chunk_meta):
        row = ["docs/auth.md", None, "JWT Config", 0.92, 3]
        errors = chunk_meta.validate_row(row)
        assert errors == []

    def test_validate_row_wrong_length(self, chunk_meta):
        row = ["docs/auth.md", 0.92]
        errors = chunk_meta.validate_row(row)
        assert len(errors) == 1
        assert "expected 5 fields" in errors[0]

    def test_validate_row_wrong_type(self, chunk_meta):
        row = [123, None, "JWT Config", 0.92, 3]  # source should be string
        errors = chunk_meta.validate_row(row)
        assert len(errors) == 1
        assert "source" in errors[0]

    def test_validate_cutdown_row(self, chunk_meta):
        row = ["docs/auth.md", 0.92]
        errors = chunk_meta.validate_row(row, mask=0b10010)
        assert errors == []

    def test_examples_valid(self, chunk_meta):
        for ex in chunk_meta.examples:
            errors = chunk_meta.validate_row(list(ex))
            assert errors == [], f"example {ex} failed: {errors}"


class TestSchemaRegistry:
    def test_load_default(self):
        reg = load_default_registry()
        assert "rag-chunk-meta:v1" in reg
        assert "rag-query-hint:v1" in reg
        assert "rag-result-summary:v1" in reg
        assert "rag-rerank-signal:v1" in reg

    def test_list(self):
        reg = load_default_registry()
        ids = reg.list()
        assert len(ids) == 4

    def test_get_missing(self):
        reg = load_default_registry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent:v1")

    def test_all_schemas_valid(self):
        """Verify all loaded schemas pass their own example validation."""
        reg = load_default_registry()
        for schema_id in reg.list():
            schema = reg.get(schema_id)
            assert schema.field_count == len(schema.fields)
            for ex in schema.examples:
                errors = schema.validate_row(list(ex))
                assert errors == [], f"{schema_id} example {ex}: {errors}"
