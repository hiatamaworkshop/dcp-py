"""Tests for shadow_level — header density control in DcpEncoder."""

import json
import pytest
from dcp_rag.core.schema import DcpSchema, load_default_registry
from dcp_rag.core.mapping import FieldMapping
from dcp_rag.core.encoder import DcpEncoder


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
    return DcpEncoder(schema=SCHEMA, mapping=MAPPING, enable_grouping=False, **kwargs)


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

        meta, text = result.groups[0][1][0]
        # Should be "source: docs/auth.md, page: 12, ..." not JSON array
        assert "source: docs/auth.md" in meta
        assert "score: 0.92" in meta
        # Should NOT be a valid JSON array
        assert not meta.startswith("[")

    def test_data_rows_unchanged_l0_to_l3(self):
        """L0-L3 should produce identical data rows (only header differs)."""
        enc = make_encoder()
        results = [enc.encode(CHUNKS, texts=TEXTS, shadow_level=i) for i in range(4)]

        # Extract data rows from each
        rows_per_level = []
        for r in results:
            rows = [meta for _, row_list in r.groups for meta, _ in row_list]
            rows_per_level.append(rows)

        for i in range(1, 4):
            assert rows_per_level[i] == rows_per_level[0], \
                f"L{i} data rows differ from L0"

    def test_grouping_works_with_shadow_levels(self):
        """$G grouping should work at all positional levels (L0-L3)."""
        enc = DcpEncoder(schema=SCHEMA, mapping=MAPPING, enable_grouping=True)
        # Use chunks with duplicate source for grouping
        chunks = [
            {"score": 0.9, "metadata": {"source": "docs/auth.md", "page": 1, "section": "A", "chunk_index": 0}},
            {"score": 0.8, "metadata": {"source": "docs/auth.md", "page": 2, "section": "B", "chunk_index": 1}},
        ]
        texts = ["text1", "text2"]

        for level in range(4):
            result = enc.encode(chunks, texts=texts, shadow_level=level)
            assert result.is_grouped, f"L{level} should still group"

    def test_level4_no_grouping(self):
        """L4 NL fallback should not use grouping."""
        enc = DcpEncoder(schema=SCHEMA, mapping=MAPPING, enable_grouping=True)
        chunks = [
            {"score": 0.9, "metadata": {"source": "docs/auth.md", "page": 1, "section": "A", "chunk_index": 0}},
            {"score": 0.8, "metadata": {"source": "docs/auth.md", "page": 2, "section": "B", "chunk_index": 1}},
        ]
        texts = ["text1", "text2"]

        result = enc.encode(chunks, texts=texts, shadow_level=4)
        assert not result.is_grouped

    def test_cutdown_with_shadow_levels(self):
        """Cutdown should work at all shadow levels."""
        enc = make_encoder()
        # Chunks with missing page and chunk_index
        sparse_chunks = [
            {"score": 0.9, "metadata": {"source": "docs/auth.md", "section": "A"}},
        ]
        texts = ["text"]

        for level in range(5):
            result = enc.encode(sparse_chunks, texts=texts, shadow_level=level)
            assert result.is_cutdown, f"L{level} should detect cutdown"