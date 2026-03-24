"""Tests for DcpEncoder — cutdown, $G grouping, batch encoding."""

import json
import pytest

from dcp_rag.core.schema import DcpSchema, load_default_registry
from dcp_rag.core.mapping import FieldMapping
from dcp_rag.core.encoder import DcpEncoder, EncodedBatch


@pytest.fixture
def registry():
    return load_default_registry()


@pytest.fixture
def pinecone_encoder(registry):
    return DcpEncoder.from_preset("pinecone", registry=registry)


@pytest.fixture
def custom_encoder(registry):
    return DcpEncoder(
        schema="rag-chunk-meta:v1",
        mapping={
            "source": "metadata.file_path",
            "page": "metadata.page_num",
            "section": "metadata.heading",
            "score": "score",
            "chunk_index": "metadata.idx",
        },
        registry=registry,
    )


# -- Sample data --

def _pinecone_chunks():
    """3 chunks from 2 sources, Pinecone format."""
    return [
        {
            "id": "1", "score": 0.92,
            "metadata": {"source": "docs/auth.md", "page": None, "section": "JWT Config", "chunk_index": 3},
        },
        {
            "id": "2", "score": 0.88,
            "metadata": {"source": "docs/auth.md", "page": None, "section": "OAuth2", "chunk_index": 4},
        },
        {
            "id": "3", "score": 0.78,
            "metadata": {"source": "api/endpoints.yaml", "page": None, "section": "/login", "chunk_index": 5},
        },
    ]


def _pinecone_texts():
    return [
        "JWT tokens expire after 24 hours.",
        "OAuth2 authorization code flow.",
        "POST /users/login accepts email and password.",
    ]


def _sparse_chunks():
    """Chunks with only source + score (page, section, chunk_index missing)."""
    return [
        {"id": "1", "score": 0.55, "metadata": {"source": "docs/billing.md"}},
        {"id": "2", "score": 0.53, "metadata": {"source": "docs/billing.md"}},
        {"id": "3", "score": 0.51, "metadata": {"source": "api/webhooks.yaml"}},
    ]


def _sparse_texts():
    return ["Billing cycles reset...", "Overage charges apply...", "Webhooks sent via POST..."]


def _unique_source_chunks():
    """All chunks from different sources — grouping should be skipped."""
    return [
        {"id": "1", "score": 0.9, "metadata": {"source": "a.md", "page": 1, "section": "A", "chunk_index": 0}},
        {"id": "2", "score": 0.8, "metadata": {"source": "b.md", "page": 2, "section": "B", "chunk_index": 1}},
        {"id": "3", "score": 0.7, "metadata": {"source": "c.md", "page": 3, "section": "C", "chunk_index": 2}},
    ]


class TestBitmaskDetection:
    def test_full_mask(self, pinecone_encoder, registry):
        chunks = _pinecone_chunks()
        resolved = [pinecone_encoder._mapping.resolve(c) for c in chunks]
        mask = pinecone_encoder.detect_mask(resolved)
        schema = registry.get("rag-chunk-meta:v1")
        # page is None for all, but it's in metadata → resolve returns None
        # source, section, score, chunk_index are present → 4 bits
        # page is None → bit not set
        assert mask == 0b10110 | 0b00001  # source + section + score + chunk_index = 10111 = 0x17
        # Wait, page is in metadata but value is None.
        # resolve_path returns None → bit not set.
        # So: source(1), page(0), section(1), score(1), chunk_index(1) = 10111
        assert mask == 0b10111

    def test_sparse_mask(self, pinecone_encoder):
        chunks = _sparse_chunks()
        resolved = [pinecone_encoder._mapping.resolve(c) for c in chunks]
        mask = pinecone_encoder.detect_mask(resolved)
        # source + score only
        assert mask == 0b10010

    def test_empty_batch(self, pinecone_encoder):
        mask = pinecone_encoder.detect_mask([])
        assert mask == 0


class TestCutdown:
    def test_no_cutdown_full_fields(self, pinecone_encoder):
        chunks = [
            {"id": "1", "score": 0.9,
             "metadata": {"source": "a.md", "page": 1, "section": "A", "chunk_index": 0}},
        ]
        result = pinecone_encoder.encode(chunks, texts=["Hello"])
        assert not result.is_cutdown
        assert result.mask == 0b11111

    def test_cutdown_sparse(self, pinecone_encoder):
        chunks = _sparse_chunks()
        texts = _sparse_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        assert result.is_cutdown
        assert result.mask == 0b10010
        assert "#12" in result.schema_id  # hex(0b10010) = 0x12

    def test_cutdown_header_contains_active_fields(self, pinecone_encoder):
        chunks = _sparse_chunks()
        texts = _sparse_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        header = json.loads(result.header)
        assert header[0] == "$S"
        assert header[2] == 2  # field count
        assert header[3] == "source"
        assert header[4] == "score"

    def test_degenerate_empty_mask(self, registry):
        """If no fields resolve at all, encoder returns empty."""
        encoder = DcpEncoder(
            schema="rag-chunk-meta:v1",
            mapping={"source": "nonexistent.path", "score": "also.missing"},
            registry=registry,
        )
        chunks = [{"id": "1", "something": "else"}]
        result = encoder.encode(chunks, texts=["text"])
        assert result.mask == 0
        assert result.header == ""


class TestGrouping:
    def test_grouped_output(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        assert result.is_grouped
        # 2 groups: docs/auth.md (2 chunks), api/endpoints.yaml (1 chunk)
        assert len(result.groups) == 2

        # First group: docs/auth.md
        g0_header, g0_rows = result.groups[0]
        g0h = json.loads(g0_header)
        assert g0h[0] == "$G"
        assert g0h[1] == "docs/auth.md"
        assert g0h[2] == 2  # 2 chunks in group
        assert len(g0_rows) == 2

        # Second group: api/endpoints.yaml
        g1_header, g1_rows = result.groups[1]
        g1h = json.loads(g1_header)
        assert g1h[1] == "api/endpoints.yaml"
        assert g1h[2] == 1

    def test_no_grouping_unique_sources(self, pinecone_encoder):
        chunks = _unique_source_chunks()
        texts = ["A text", "B text", "C text"]
        result = pinecone_encoder.encode(chunks, texts=texts)
        assert not result.is_grouped

    def test_grouping_disabled(self, registry):
        encoder = DcpEncoder.from_preset(
            "pinecone", registry=registry, enable_grouping=False
        )
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = encoder.encode(chunks, texts=texts)
        assert not result.is_grouped

    def test_group_sorted_by_score(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        # docs/auth.md group: 0.92 should come before 0.88
        _, rows = result.groups[0]
        scores = []
        for meta_json, _ in rows:
            meta = json.loads(meta_json)
            # With cutdown (page is None for all → dropped), grouped row fields are:
            # section, score, chunk_index (source in $G header, page dropped)
            scores.append(meta[1])  # score at position 1 (after section)
        assert scores == sorted(scores, reverse=True)

    def test_source_not_in_grouped_rows(self, pinecone_encoder):
        """source field should be in $G header, not in per-chunk rows."""
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        _, rows = result.groups[0]
        for meta_json, _ in rows:
            meta = json.loads(meta_json)
            # Cutdown active: page is None for all → dropped from mask
            # Grouped removes source → row has: section, score, chunk_index = 3 fields
            assert len(meta) == 3
            # Verify source string is not in the row values
            assert "docs/auth.md" not in [str(v) for v in meta]


class TestEncodedBatch:
    def test_to_lines(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        lines = result.to_lines()
        # First line: $S header
        assert lines[0].startswith('["$S"')
        # Should contain $G lines
        g_lines = [l for l in lines if l.startswith('["$G"')]
        assert len(g_lines) == 2

    def test_to_string(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        s = result.to_string()
        assert '"$S"' in s
        assert '"$G"' in s
        assert "JWT tokens expire" in s

    def test_meta_only_lines(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        meta_lines = result.meta_only_lines()
        # Should not contain text
        for line in meta_lines:
            assert "JWT tokens expire" not in line


class TestEncodeMetadata:
    def test_single_chunk(self, pinecone_encoder):
        chunk = {
            "score": 0.92,
            "metadata": {"source": "docs/auth.md", "page": None, "section": "JWT", "chunk_index": 3},
        }
        result = pinecone_encoder.encode_metadata(chunk)
        assert "_dcp" in result
        assert "_dcp_schema" in result
        # page is None → cutdown
        assert result["_dcp_schema"] == "rag-chunk-meta:v1#17"  # 0b10111

    def test_sparse_chunk(self, pinecone_encoder):
        chunk = {"score": 0.5, "metadata": {"source": "test.md"}}
        result = pinecone_encoder.encode_metadata(chunk)
        assert result["_dcp"] == ["test.md", 0.5]
        assert "#12" in result["_dcp_schema"]

    def test_empty_chunk(self):
        """Chunk with no resolvable fields returns empty dict."""
        encoder = DcpEncoder(
            schema="rag-chunk-meta:v1",
            mapping={"source": "nonexistent"},
        )
        result = encoder.encode_metadata({"something": "else"})
        assert result == {}


class TestEmptyBatch:
    def test_empty_chunks(self, pinecone_encoder):
        result = pinecone_encoder.encode([], texts=[])
        assert result.header == ""
        assert result.groups == []
        assert result.mask == 0


class TestFromPreset:
    def test_all_presets_load(self, registry):
        for preset in ["pinecone", "qdrant", "weaviate", "chroma", "milvus"]:
            encoder = DcpEncoder.from_preset(preset, registry=registry)
            assert encoder is not None

    def test_unknown_preset(self, registry):
        with pytest.raises(KeyError, match="no preset"):
            DcpEncoder.from_preset("unknown_db", registry=registry)

    def test_preset_with_overrides(self, registry):
        encoder = DcpEncoder.from_preset(
            "pinecone",
            overrides={"section": "metadata.heading"},
            registry=registry,
        )
        chunk = {
            "score": 0.9,
            "metadata": {"source": "x.md", "heading": "Setup", "chunk_index": 0},
        }
        result = encoder.encode_metadata(chunk)
        assert "Setup" in result["_dcp"]


class TestTextKey:
    def test_text_key_extraction(self, registry):
        encoder = DcpEncoder.from_preset(
            "pinecone", registry=registry, text_key="metadata.text"
        )
        chunks = [
            {"score": 0.9, "metadata": {"source": "a.md", "text": "Hello world", "chunk_index": 0}},
        ]
        result = encoder.encode(chunks)
        lines = result.to_lines()
        assert any("Hello world" in l for l in lines)

    def test_no_text_key_no_texts_raises(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        with pytest.raises(ValueError, match="text_key"):
            pinecone_encoder.encode(chunks)

    def test_length_mismatch_raises(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        with pytest.raises(ValueError, match="mismatch"):
            pinecone_encoder.encode(chunks, texts=["only one"])
