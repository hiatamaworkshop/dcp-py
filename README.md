# DCP-RAG

Data Cost Protocol encoder for system→AI data injection. Converts structured data (metadata, signals, logs) into schema-typed positional arrays before they enter the LLM context window.

RAG pipelines are the primary use case, but the core is domain-agnostic — any `dict → positional array` transformation works.

## Problem

Every time structured data enters an LLM context window as natural language, you pay for redundant key names, labels, and formatting that the model doesn't need.

```
RAG:     "Source: docs/auth.md\nPage: 12\nScore: 0.92\n..."   → keys repeated per chunk
Logs:    "Error in auth-service at 2024-03-24: timeout"        → parsing requires inference
API:     {"status": 200, "latency_ms": 42, "endpoint": "/v1"}  → keys are pure overhead
```

## Solution

Define a schema once. Write data by position. Strip everything the consumer doesn't need.

```
Schema:  ["$S","rag-chunk-meta:v1",5,"source","page","section","score","chunk_index"]
Data:    ["docs/auth.md",12,"JWT Config",0.92,3]
```

Metadata reduction: **40-60%**. Total RAG prompt reduction: **10-15%** (chunk text is untouched). The real gain is cleaner context — same window fits more useful data, improving response quality.

## Quick Start: RAG

```python
from dcp_rag.core.encoder import DcpEncoder

# Preset — 1 line for supported Vector DBs
encoder = DcpEncoder.from_preset("pinecone")

# Preset + overrides — custom metadata field names
encoder = DcpEncoder.from_preset("qdrant", overrides={
    "section": "payload.heading_text",
})

# Full custom mapping — any DB, any metadata structure
encoder = DcpEncoder(schema="rag-chunk-meta:v1", mapping={
    "source": "metadata.file_path",
    "page":   "metadata.page_num",
    "section": "metadata.heading",
    "score":  "score",
    "chunk_index": "metadata.idx",
})

# Encode search results
result = encoder.encode(search_results, texts=chunk_texts)
print(result.to_string())
```

### Framework Integration

```python
# LlamaIndex — node_postprocessor
query_engine = index.as_query_engine(
    node_postprocessors=[DcpNodePostprocessor.from_preset("pinecone")]
)

# LangChain — LCEL pipe
chain = retriever | DcpRunnable.from_preset("qdrant") | prompt | llm

# Haystack — pipeline component
pipeline.add_component("dcp", DcpComponent.from_preset("weaviate"))
pipeline.connect("retriever", "dcp")
pipeline.connect("dcp", "prompt_builder")
```

## Beyond RAG: Universal DCP Encoding

The core engine (`DcpSchema`, `FieldMapping`, `DcpEncoder`) has zero RAG-specific code. Any structured data going into an LLM context window can be DCP-encoded by writing a schema JSON and a field mapping.

### Logs → LLM

```python
from dcp_rag.core.schema import DcpSchema
from dcp_rag.core.encoder import DcpEncoder

# Define schema (or load from JSON file)
schema = DcpSchema.from_dict({
    "$dcp": "schema",
    "id": "log-entry:v1",
    "fields": ["level", "service", "timestamp", "error_code"],
    "fieldCount": 4,
    "types": {
        "level": {"type": "string", "enum": ["debug", "info", "warn", "error", "fatal"]},
        "service": {"type": "string"},
        "timestamp": {"type": "number"},
        "error_code": {"type": ["string", "null"]}
    }
})

encoder = DcpEncoder(
    schema=schema,
    mapping={
        "level": "level",
        "service": "service_name",
        "timestamp": "ts",
        "error_code": "error.code",
    },
    group_key="service",   # $G groups by service
    text_key="message",    # log body
)

result = encoder.encode(log_entries)
# Before: "Error in auth-service at 1711284600: connection timeout (E_TIMEOUT)"
# After:  ["error","auth-service",1711284600,"E_TIMEOUT"]
#          connection timeout
```

### API Responses → LLM

```python
schema = DcpSchema.from_dict({
    "$dcp": "schema",
    "id": "api-response:v1",
    "fields": ["status", "latency_ms", "endpoint", "method"],
    "fieldCount": 4,
    "types": {
        "status": {"type": "number"},
        "latency_ms": {"type": "number"},
        "endpoint": {"type": "string"},
        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]}
    }
})

encoder = DcpEncoder(
    schema=schema,
    mapping={
        "status": "response.status_code",
        "latency_ms": "metrics.duration_ms",
        "endpoint": "request.path",
        "method": "request.method",
    },
    group_key="endpoint",
)

result = encoder.encode(api_calls)
# ["$S","api-response:v1",4,"status","latency_ms","endpoint","method"]
# ["$G","/v1/users",3]
# [200,42,"GET"]
# [201,89,"POST"]
# [200,38,"GET"]
```

### Writing Custom Schemas

Create a JSON file in `schemas/`:

```json
{
  "$dcp": "schema",
  "id": "your-domain:v1",
  "fields": ["field1", "field2", "field3"],
  "fieldCount": 3,
  "types": {
    "field1": { "type": "string" },
    "field2": { "type": "number", "min": 0, "max": 1 },
    "field3": { "type": ["string", "null"] }
  }
}
```

Load it:

```python
schema = DcpSchema.from_file("schemas/your-domain.v1.json")
encoder = DcpEncoder(schema=schema, mapping={"field1": "src.key1", ...})
```

The encoder handles the rest: `$S` header generation, bitmask cutdown for missing fields, `$G` source grouping, and positional array encoding.

## Design Principles

1. **Data is untouched** — only the representation changes. Values are as-is, structure becomes positional
2. **Schema-typed positional arrays** — no parsing ambiguity, no field name overhead
3. **1-line integration** — DB presets for zero-config; custom schema + mapping for any domain
4. **LLM boundary only** — encode at the point where data enters the context window, not before
5. **Cutdown over null-fill** — missing fields are omitted via bitmask, not padded with nulls
6. **Domain-agnostic core** — RAG presets and adapters are configuration, not code

## Architecture

```
Layer 0: DcpSchema       ← Schema definition (fields, types, validation)     [universal]
Layer 1: FieldMapping     ← source key → DCP positional index                [universal]
Layer 2: Preset           ← Per-source defaults (Pinecone, Qdrant, ...)      [domain-specific]
Layer 3: Adapter          ← Per-framework (LlamaIndex, LangChain, ...)       [domain-specific]
```

Layers 0-1 are the universal engine. Layers 2-3 are RAG configuration — add your own presets/adapters for other domains.

```
dcp-rag/
  core/
    schema.py        ← Schema loader + validator + bitmask ops (Layer 0)
    mapping.py       ← FieldMapping: dot-notation path resolver (Layer 1)
    encoder.py       ← DcpEncoder: $S header, $G grouping, cutdown (Layer 0+1)
    presets/         ← DB presets: pinecone, qdrant, weaviate, chroma, milvus (Layer 2)
  adapters/
    llamaindex.py    ← DcpNodePostprocessor (Layer 3)
    langchain.py     ← DcpRunnable (Layer 3)
    haystack.py      ← DcpComponent (Layer 3)
    azure.py         ← Azure AI Search Custom Skill (Layer 3)
  schemas/           ← DCP schema definitions (RAG + custom)
```

### Where DCP sits in the pipeline

```
search → reranker → filter → compressor → [★ DCP encoder ★] → LLM
                                            ↑
                              only point where tokens cost money
```

Upstream stages (reranker, filter, compressor) are rule-based or small models — no token cost.
Original metadata is preserved for them. DCP is additive:

```python
node.metadata = {
    "source": "docs/auth.md",          # ← untouched for upstream stages
    "page": 12,
    "_dcp": ["docs/auth.md", 12, "JWT Config", 0.92, 3],
    "_dcp_schema": "rag-chunk-meta:v1"  # ← DCP-aware prompt builder uses this
}
```

## Vector DB Presets

| DB | Response structure | Preset handles |
|----|-------------------|----------------|
| **Pinecone** | `{ id, score, metadata: { ... } }` | score mapping, metadata.* passthrough |
| **Qdrant** | `{ id, score, payload: { ... } }` | score mapping, payload.* passthrough |
| **Weaviate** | `{ _additional: { score }, properties: { ... } }` | _additional.score, properties.* |
| **Chroma** | `{ ids, distances, metadatas, documents }` | distance→score conversion, metadatas.* |
| **Milvus** | `{ id, distance, entity: { ... } }` | distance→score conversion, entity.* |

## Schemas

RAG-specific DCP schemas (planned):

| Schema | Purpose | Fields |
|--------|---------|--------|
| `rag-chunk-meta:v1` | Chunk metadata | source, page, section, score, chunk_index |
| `rag-query-hint:v1` | Inter-stage query signal | intent, domain, detail, urgency |
| `rag-result-summary:v1` | Stage result summary | found, count, domain, avg_score |
| `rag-rerank-signal:v1` | Reranking output | chunk_id, original_rank, new_rank, boost_reason |

## Key Features

| Feature | What it does |
|---------|-------------|
| **$S header** | Schema declaration inline with data — consumer knows the structure in ~5 tokens |
| **Bitmask cutdown** | Missing fields are omitted, not null-padded. Sparse data gets smaller, not wider |
| **$G grouping** | Repeated source values are factored out into group headers. 10 chunks from 3 sources → 3 groups |
| **Schema validation** | `validate_row()` checks types, enums, ranges. Catches errors at encode time |
| **Preset system** | 1-line setup for known data sources. Override individual fields without rewriting the mapping |

## Status

v0.1.0 — Core encoder, 5 DB presets, 4 framework adapters. Unit-tested (72 tests) and integration-tested against live Qdrant. Metadata token reduction varies by data structure: **40-60% on typical RAG metadata**, up to 69% on metadata-dense knowledge graphs (engram). Chunk text is unchanged — total prompt reduction depends on your metadata-to-content ratio (typically 10-15%).

## Related

- [engram](https://github.com/hiatamaworkshop/engram) — Origin project where DCP was developed
- [Data Cost Protocol](https://engram-docs.vercel.app/engram/data-cost-protocol) — Full DCP specification

## License

Apache License 2.0

---

Designed by Hiatama Workshop