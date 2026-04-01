"""Tests for shadow_level — header density control in DcpEncoder."""

import json
import pytest
from dcp_py.core.schema import DcpSchema, load_default_registry
from dcp_py.core.mapping import FieldMapping
from dcp_py.core.encoder import DcpEncoder


SCHEMA = load_default_registry().get("rag-chunk-meta:v1")
MAPPING = FieldMapping(
    schema_id="rag-chunk-meta:v1",
    paths={
        "source": "metadata.source",
        "page": "metadata.page",
        "section": "metadata.section",
        "score": "score",
        "chunk_index": "metadata.chunk_index",
    },
)

CHUNKS = [
    {
        "score": 0.92,
        "metadata": {"source": "docs/auth.md", "page": 12, "section": "JWT Config", "chunk_index": 3},
    },
    {
        "score": 0.87,
        "metadata": {"source": "docs/api.md", "page": 5, "section": "Rate Limiting", "chunk_index": 1},
    },
]
TEXTS = ["auth content", "api content"]


def make_encoder(**kwargs):
    return DcpEncoder(schema=SCHEMA, mapping=MAPPING, **kwargs)


class TestShadowLevel:

    def test_level0_fields_only(self):
        """L0: just field names, no protocol markers."""
        enc = make_encoder()
        result = enc.encode(CHUNKS, texts=TEXTS, shadow_level=0)
        header = json.loads(result.header)

        assert "$S" not in header
        assert "rag-chunk-meta" not in str(header)
        assert "source" in header
        assert "score" in header
        assert len(header) == 5  # 5 field names

    def test_level1_with_schema_id(self):
        """L1: $S + schema ID + field names, no field count."""
        enc = make_encoder()
        result = enc.encode(CHUNKS, texts=TEXTS, shadow_level=1)
        header = json.loads(result.header)

        assert header[0] == "$S"
        assert "rag-chunk-meta:v1" in header[1]
        assert "source" in header
        # No field count number between schema ID and field names
        assert isinstance(header[2], str)  # field name, not int

    def test_level2_full_protocol(self):
        """L2: $S + ID + field count + field names (default, original behavior)."""
        enc = make_encoder()
        result = enc.encode(CHUNKS, texts=TEXTS, shadow_level=2)
        header = json.loads(result.header)

        assert header[0] == "$S"
        assert header[1] == "rag-chunk-meta:v1"
        assert header[2] == 5  # field count
        assert header[3] == "source"

    def test_level2_is_default(self):
        """Default shadow_level should be 2."""
        enc = make_encoder()
        default_result = enc.encode(CHUNKS, texts=TEXTS)
        explicit_result = enc.encode(CHUNKS, texts=TEXTS, shadow_level=2)
        assert default_result.header == explicit_result.header

    def test_level3_full_schema(self):
        """L3: complete schema definition as JSON object."""
        enc = make_encoder()
        result = enc.encode(CHUNKS, texts=TEXTS, shadow_level=3)
        header = json.loads(result.header)

        assert isinstance(header, dict)
        assert header["$dcp"] == "schema"
        assert header["id"] == "rag-chunk-meta:v1"
        assert "fields" in header
        assert "types" in header
        assert "fieldCount" in header

    def test_level4_nl_fallback_header(self):
        """L4: header is natural language field descriptions."""
        enc = make_encoder()
        result = enc.encode(CHUNKS, texts=TEXTS, shadow_level=4)

        # Header should be a plain string, not JSON
        assert result.header.startswith("Fields:")
        assert "source" in result.header

    def test_level4_nl_fallback_rows(self):
        """L4: data rows are key-value strings, not positional arrays."""
        enc = make_encoder()
        result = enc.encode(CHUNKS, texts=TEXTS, shadow_level=4)

        meta, text = result.rows[0]
        assert "source: docs/auth.md" in meta
        assert "score: 0.92" in meta
        assert not meta.startswith("[")

    def test_data_rows_unchanged_l0_to_l3(self):
        """L0-L3 should produce identical data rows (only header differs)."""
        enc = make_encoder()
        results = [enc.encode(CHUNKS, texts=TEXTS, shadow_level=i) for i in range(4)]

        rows_per_level = [[meta for meta, _ in r.rows] for r in results]
        for i in range(1, 4):
            assert rows_per_level[i] == rows_per_level[0], \
                f"L{i} data rows differ from L0"

    def test_cutdown_with_shadow_levels(self):
        """Cutdown should work at all shadow levels."""
        enc = make_encoder()
        sparse_chunks = [
            {"score": 0.9, "metadata": {"source": "docs/auth.md", "section": "A"}},
        ]
        texts = ["text"]

        for level in range(5):
            result = enc.encode(sparse_chunks, texts=texts, shadow_level=level)
            assert result.is_cutdown, f"L{level} should detect cutdown"

    def test_no_grouping_in_any_level(self):
        """$G should never appear at any shadow level."""
        enc = make_encoder()
        for level in range(5):
            result = enc.encode(CHUNKS, texts=TEXTS, shadow_level=level)
            assert '"$G"' not in result.to_string()