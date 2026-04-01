"""Integration test: dcp-py against live engram Qdrant instance.

Connects to engram's Qdrant (localhost:6333, collection "engram"),
reads existing nodes, encodes them with DCP, and verifies output.

Requires: engram Qdrant running (docker compose up from engram/)
Skip if Qdrant is not available.

engram UpperLayerPointPayload structure:
  {
    id: string,
    score: number,          # from Qdrant search
    payload: {
      summary: string,      # knowledge headline (→ chunk text)
      content: string,      # detailed explanation
      tags: string[],
      projectId: string,    # project scope
      source: string,       # "mcp-ingest" (ingestion source, not doc source)
      trigger: string,      # "session-end", "milestone", etc.
      status: "recent" | "fixed",
      weight: number,       # survival score
      hitCount: number,
      ingestedAt: number,   # Unix ms
      native?: unknown[],   # DCP native (new data)
      schema?: string,      # DCP schema ID
      index?: string,       # human-readable restore key
    }
  }
"""

import json
import os
import urllib.request
import urllib.error
import pytest

from dcp_py.core.schema import DcpSchema, SchemaRegistry, load_default_registry
from dcp_py.core.mapping import FieldMapping
from dcp_py.core.encoder import DcpEncoder


QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "engram"


def qdrant_available() -> bool:
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/healthz", method="GET")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def qdrant_scroll(limit: int = 20, project_filter: str | None = None) -> list[dict]:
    """Scroll points from engram collection via Qdrant REST API."""
    body: dict = {"limit": limit, "with_payload": True, "with_vector": False}
    if project_filter:
        body["filter"] = {
            "must": [{"key": "projectId", "match": {"value": project_filter}}]
        }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        result = json.loads(resp.read())
    return result["result"]["points"]


def qdrant_search(vector: list[float], limit: int = 5) -> list[dict]:
    """Search engram collection with a dummy vector — just to get score field."""
    body = {"vector": vector, "limit": limit, "with_payload": True}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        result = json.loads(resp.read())
    return result["result"]


def qdrant_collection_info() -> dict:
    """Get collection info to determine vector dimension."""
    req = urllib.request.Request(f"{QDRANT_URL}/collections/{COLLECTION}", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        result = json.loads(resp.read())
    return result["result"]


# -- Custom schema for engram nodes --

ENGRAM_SCHEMA_DEF = {
    "$dcp": "schema",
    "id": "engram-node-meta:v1",
    "description": "engram knowledge node metadata for DCP encoding",
    "fields": ["projectId", "source", "trigger", "status", "weight", "tags"],
    "fieldCount": 6,
    "types": {
        "projectId": {"type": "string", "description": "Project scope"},
        "source": {"type": "string", "description": "Ingestion source"},
        "trigger": {"type": "string", "description": "What triggered ingestion"},
        "status": {"type": "string", "enum": ["recent", "fixed"]},
        "weight": {"type": "number", "min": -100, "max": 100},
        "tags": {"type": "string", "description": "Comma-joined tags"},
    },
    "examples": [
        ["engram", "mcp-ingest", "milestone", "recent", 0.5, "howto,docker"],
        ["dcp-rag", "mcp-ingest", "design-decision", "fixed", 2.0, "design,architecture"],
    ],
}

ENGRAM_MAPPING = FieldMapping(
    schema_id="engram-node-meta:v1",
    paths={
        "projectId": "payload.projectId",
        "source": "payload.source",
        "trigger": "payload.trigger",
        "status": "payload.status",
        "weight": "payload.weight",
        "tags": "payload.tags_joined",  # join tags pre-encoding
    },
)


skip_no_qdrant = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant not running at localhost:6333",
)


@skip_no_qdrant
class TestEngramQdrantIntegration:
    """Integration tests against live engram Qdrant data."""

    def test_qdrant_health(self):
        assert qdrant_available()

    def test_scroll_returns_data(self):
        points = qdrant_scroll(limit=5)
        assert len(points) > 0
        p = points[0]
        assert "id" in p
        assert "payload" in p
        payload = p["payload"]
        assert "summary" in payload
        assert "projectId" in payload

    def test_encode_scroll_results(self):
        """Encode scrolled engram nodes with custom schema + mapping."""
        points = qdrant_scroll(limit=10)
        assert len(points) > 0

        schema = DcpSchema.from_dict(ENGRAM_SCHEMA_DEF)

        for p in points:
            tags = p["payload"].get("tags", [])
            p["payload"]["tags_joined"] = ",".join(tags) if tags else None

        encoder = DcpEncoder(
            schema=schema,
            mapping=ENGRAM_MAPPING,
            text_key="payload.summary",
        )

        result = encoder.encode(points)

        assert result.header != ""
        assert result.mask > 0
        header = json.loads(result.header)
        assert header[0] == "$S"
        assert "engram-node-meta" in header[1]

        # Absent values must be '-', not None
        for row_json, _ in result.rows:
            row = json.loads(row_json)
            assert None not in row

        print("\n--- DCP Encoded engram nodes ---")
        for line in result.to_lines()[:20]:
            print(line)

    def test_search_with_dcp(self):
        """Search with a zero vector to get scored results, then encode."""
        info = qdrant_collection_info()
        dim = info["config"]["params"]["vectors"]["size"]

        zero_vec = [0.0] * dim
        search_results = qdrant_search(zero_vec, limit=5)
        if not search_results:
            pytest.skip("No search results returned")

        schema = DcpSchema.from_dict(ENGRAM_SCHEMA_DEF)

        for r in search_results:
            tags = r["payload"].get("tags", [])
            r["payload"]["tags_joined"] = ",".join(tags) if tags else None

        encoder = DcpEncoder(
            schema=schema,
            mapping=ENGRAM_MAPPING,
            text_key="payload.summary",
        )

        result = encoder.encode(search_results)
        assert result.header != ""

        print("\n--- DCP Encoded search results ---")
        for line in result.to_lines()[:15]:
            print(line)

    def test_token_comparison(self):
        """Compare NL vs DCP token cost on real engram data."""
        points = qdrant_scroll(limit=10)
        if not points:
            pytest.skip("No points in collection")

        # NL format
        nl_lines = []
        for i, p in enumerate(points):
            payload = p["payload"]
            nl_lines.append(f"[Node {i+1}]")
            nl_lines.append(f"Project: {payload.get('projectId', '?')}")
            nl_lines.append(f"Summary: {payload.get('summary', '?')}")
            nl_lines.append(f"Tags: {', '.join(payload.get('tags', []))}")
            nl_lines.append(f"Status: {payload.get('status', '?')}")
            nl_lines.append(f"Weight: {payload.get('weight', '?')}")
            if payload.get("content"):
                nl_lines.append(f"Content: {payload['content']}")
            nl_lines.append("")
        nl_text = "\n".join(nl_lines)

        # DCP format
        schema = DcpSchema.from_dict(ENGRAM_SCHEMA_DEF)
        for p in points:
            tags = p["payload"].get("tags", [])
            p["payload"]["tags_joined"] = ",".join(tags) if tags else None

        encoder = DcpEncoder(
            schema=schema,
            mapping=ENGRAM_MAPPING,
            text_key="payload.summary",
        )
        result = encoder.encode(points)
        dcp_text = result.to_string()

        # Rough token estimate (1 token ≈ 3.8 chars)
        nl_tokens = len(nl_text) / 3.8
        dcp_tokens = len(dcp_text) / 3.8
        reduction = (1 - dcp_tokens / nl_tokens) * 100

        print(f"\n--- Token comparison ({len(points)} nodes) ---")
        print(f"NL:  {nl_tokens:.0f} tokens ({len(nl_text)} chars)")
        print(f"DCP: {dcp_tokens:.0f} tokens ({len(dcp_text)} chars)")
        print(f"Reduction: {reduction:.0f}%")

        assert dcp_tokens < nl_tokens, "DCP should use fewer tokens than NL"