"""Tests for DcpEncoder — cutdown, '-' absent values, batch encoding, DataFrame."""

import json
import pytest

from dcp_py.core.schema import DcpSchema, load_default_registry
from dcp_py.core.mapping import FieldMapping
from dcp_py.core.encoder import DcpEncoder, EncodedBatch


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


class TestBitmaskDetection:
    def test_full_mask(self, pinecone_encoder, registry):
        chunks = _pinecone_chunks()
        resolved = [pinecone_encoder._mapping.resolve(c) for c in chunks]
        mask = pinecone_encoder.detect_mask(resolved)
        # page is None for all → bit not set
        # source, section, score, chunk_index present → 0b10111
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
        assert "#12" in result.schema_id

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
        encoder = DcpEncoder(
            schema="rag-chunk-meta:v1",
            mapping={"source": "nonexistent.path", "score": "also.missing"},
            registry=registry,
        )
        chunks = [{"id": "1", "something": "else"}]
        result = encoder.encode(chunks, texts=["text"])
        assert result.mask == 0
        assert result.header == ""


class TestAbsentValues:
    """Absent (None) fields must appear as '-', not null."""

    def test_none_field_becomes_dash(self, pinecone_encoder):
        """page is None in one row, present in other → '-' in the None row."""
        chunks = [
            {"id": "1", "score": 0.9,
             "metadata": {"source": "a.md", "page": None, "section": "A", "chunk_index": 0}},
            {"id": "2", "score": 0.8,
             "metadata": {"source": "b.md", "page": 3, "section": "B", "chunk_index": 1}},
        ]
        result = pinecone_encoder.encode(chunks, texts=["text1", "text2"])
        row = json.loads(result.rows[0][0])
        assert "-" in row
        assert None not in row

    def test_sparse_rows_have_no_null(self, pinecone_encoder):
        chunks = _sparse_chunks()
        texts = _sparse_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        for row_json, _ in result.rows:
            row = json.loads(row_json)
            assert None not in row

    def test_encode_metadata_none_becomes_dash(self, pinecone_encoder):
        chunk = {
            "score": 0.92,
            "metadata": {"source": "docs/auth.md", "page": None, "section": "JWT", "chunk_index": 3},
        }
        result = pinecone_encoder.encode_metadata(chunk)
        assert None not in result["_dcp"]
        assert "-" not in result["_dcp"]  # page dropped by cutdown (all None → bit 0)


class TestEncodedBatch:
    def test_to_lines_with_text(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        lines = result.to_lines()
        assert lines[0].startswith('["$S"')
        # each row has a text line after it
        assert any("JWT tokens expire" in l for l in lines)

    def test_to_lines_without_text(self, pinecone_encoder):
        chunks = _sparse_chunks()
        result = pinecone_encoder.encode(chunks)  # no texts
        lines = result.to_lines()
        assert lines[0].startswith('["$S"')
        # no text lines — only header + rows
        assert len(lines) == 1 + len(chunks)

    def test_to_string(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        s = result.to_string()
        assert '"$S"' in s
        assert "JWT tokens expire" in s

    def test_meta_only_lines(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        meta_lines = result.meta_only_lines()
        for line in meta_lines:
            assert "JWT tokens expire" not in line

    def test_no_grouping_in_output(self, pinecone_encoder):
        """$G should never appear — grouping is removed."""
        chunks = _pinecone_chunks()
        texts = _pinecone_texts()
        result = pinecone_encoder.encode(chunks, texts=texts)
        output = result.to_string()
        assert '"$G"' not in output


class TestEncodeMetadata:
    def test_single_chunk(self, pinecone_encoder):
        chunk = {
            "score": 0.92,
            "metadata": {"source": "docs/auth.md", "page": None, "section": "JWT", "chunk_index": 3},
        }
        result = pinecone_encoder.encode_metadata(chunk)
        assert "_dcp" in result
        assert "_dcp_schema" in result
        # page is None → cutdown (bit not set)
        assert result["_dcp_schema"] == "rag-chunk-meta:v1#17"

    def test_sparse_chunk(self, pinecone_encoder):
        chunk = {"score": 0.5, "metadata": {"source": "test.md"}}
        result = pinecone_encoder.encode_metadata(chunk)
        assert result["_dcp"] == ["test.md", 0.5]
        assert "#12" in result["_dcp_schema"]

    def test_empty_chunk(self):
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
        assert result.rows == []
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

    def test_length_mismatch_raises(self, pinecone_encoder):
        chunks = _pinecone_chunks()
        with pytest.raises(ValueError, match="mismatch"):
            pinecone_encoder.encode(chunks, texts=["only one"])

    def test_no_text_no_text_key_produces_empty_text(self, pinecone_encoder):
        """Without texts= or text_key, encode() still works — rows have empty text."""
        chunks = _sparse_chunks()
        result = pinecone_encoder.encode(chunks)
        assert len(result.rows) == len(chunks)
        for _, text in result.rows:
            assert text == ""


class TestDataFrame:
    def test_from_dataframe_basic(self):
        pytest.importorskip("pandas")
        import pandas as pd

        df = pd.DataFrame([
            {"title": "Intro to DCP", "score": 0.92, "source": "docs/dcp.md"},
            {"title": "Shadow System", "score": 0.85, "source": "docs/shadow.md"},
            {"title": "Encoder Guide", "score": 0.78, "source": "docs/encoder.md"},
        ])
        encoder, batch = DcpEncoder.from_dataframe(df, domain="search-result")
        assert batch.header != ""
        assert len(batch.rows) == 3
        # No None in any row
        for row_json, _ in batch.rows:
            row = json.loads(row_json)
            assert None not in row

    def test_from_dataframe_exclude(self):
        pytest.importorskip("pandas")
        import pandas as pd

        df = pd.DataFrame([
            {"title": "Doc A", "score": 0.9, "internal_id": "abc123"},
            {"title": "Doc B", "score": 0.8, "internal_id": "def456"},
        ])
        encoder, batch = DcpEncoder.from_dataframe(
            df, domain="docs", exclude=["internal_id"]
        )
        header = json.loads(batch.header)
        field_names = header[3:]  # after $S, id, count
        assert "internal_id" not in field_names

    def test_from_dataframe_schema_id(self):
        pytest.importorskip("pandas")
        import pandas as pd

        df = pd.DataFrame([{"x": 1, "y": 2}])
        encoder, batch = DcpEncoder.from_dataframe(df, domain="my-data", version=2)
        assert "my-data:v2" in batch.schema_id