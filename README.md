# dcp-py

Data Cost Protocol for Python — universal `dict → positional array` encoder for LLM context compression.

Any structured data flowing into an LLM context window can be DCP-encoded: RAG chunk metadata, SQL query results, log streams, API responses, sensor readings. Schema once, position always, keys never repeated.

## Problem

Every time structured data enters an LLM as natural language, you pay for redundant key names that the model doesn't need.

```
RAG:  "Source: docs/auth.md\nPage: 12\nScore: 0.92\n..."   → keys repeated per chunk
SQL:  {"id":1,"name":"Alice","dept":"Eng","salary":90000}  → keys at every row
Logs: "Error in auth-service at 2024-03-24: timeout"       → parsing requires inference
```

## Solution

Define a schema once. Write data by position. Strip everything the consumer doesn't need.

```
Schema:  ["$S","rag-chunk-meta:v1",5,"source","page","section","score","chunk_index"]
Data:    ["docs/auth.md","-","-",0.92,3]
```

Absent fields use `"-"` (single token, unambiguous). Fields absent across the entire batch are dropped via bitmask cutdown.

Metadata reduction: **40-60%**. Total RAG prompt reduction: **10-15%** (chunk text is untouched).

## Quick Start

### RAG / Vector DB

```python
from dcp_py.core.encoder import DcpEncoder

# 1-line preset for supported Vector DBs
encoder = DcpEncoder.from_preset("pinecone")

# Preset + custom field path overrides
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
batch = encoder.encode(search_results, texts=chunk_texts)
print(batch.to_string())
```

### pandas DataFrame

```python
from dcp_py.core.encoder import DcpEncoder

encoder, batch = DcpEncoder.from_dataframe(df, domain="query-result")
print(batch.to_string())
# ["$S","query-result:v1",3,"name","dept","salary"]
# ["Alice","Eng",90000]
# ["Bob","Sales",75000]

# Exclude columns
encoder, batch = DcpEncoder.from_dataframe(
    df, domain="query-result", exclude=["internal_id", "embedding"]
)
```

### Framework Integration

```python
# LlamaIndex — node_postprocessor
from dcp_py.adapters.llamaindex import DcpNodePostprocessor

query_engine = index.as_query_engine(
    node_postprocessors=[DcpNodePostprocessor.from_preset("pinecone")]
)

# LangChain — LCEL pipe
from dcp_py.adapters.langchain import DcpRunnable

chain = retriever | DcpRunnable.from_preset("qdrant") | prompt | llm

# Haystack — pipeline component
from dcp_py.adapters.haystack import DcpComponent

pipeline.add_component("dcp", DcpComponent.from_preset("weaviate"))
pipeline.connect("retriever", "dcp")
pipeline.connect("dcp", "prompt_builder")
```

## Any Domain

The core (`DcpSchema`, `FieldMapping`, `DcpEncoder`) has zero RAG-specific code. Schema + mapping = encoder for anything.

### Log streams

```python
from dcp_py.core.schema import DcpSchema
from dcp_py.core.encoder import DcpEncoder

schema = DcpSchema.from_dict({
    "$dcp": "schema",
    "id": "log-entry:v1",
    "fields": ["ts", "level", "service", "msg"],
    "fieldCount": 4,
    "types": {
        "ts":      {"type": "number"},
        "level":   {"type": "string", "enum": ["debug", "info", "warn", "error"]},
        "service": {"type": "string"},
        "msg":     {"type": "string"},
    }
})

encoder = DcpEncoder(schema=schema, mapping={
    "ts":      "timestamp",
    "level":   "level",
    "service": "service_name",
    "msg":     "message",
})

batch = encoder.encode(log_entries)
# ["$S","log-entry:v1",4,"ts","level","service","msg"]
# [1711284600,"error","auth","connection timeout"]
```

### SQL / API results

```python
# DataFrame (schema auto-inferred from columns)
encoder, batch = DcpEncoder.from_dataframe(cursor_df, domain="sales-report")

# Manual schema for fixed API shape
schema = DcpSchema.from_dict({
    "$dcp": "schema",
    "id": "api-response:v1",
    "fields": ["status", "latency_ms", "endpoint", "method"],
    "fieldCount": 4,
    "types": {
        "status":     {"type": "number"},
        "latency_ms": {"type": "number"},
        "endpoint":   {"type": "string"},
        "method":     {"type": "string", "enum": ["GET","POST","PUT","DELETE"]},
    }
})
```

## Absent Values

Fields with `None` values appear as `"-"` in the positional array — single token, unambiguous, log/CSV tradition.

```python
# page is None → encoded as "-"
["docs/auth.md", "-", "JWT Config", 0.92, 3]
```

Fields absent across the **entire batch** are dropped via bitmask cutdown — they don't appear in the row at all:

```python
# Only source + score present in all records → 2-field cutdown schema
["$S","rag-chunk-meta:v1#12",2,"source","score"]
["docs/auth.md",0.92]
```

## Shadow Levels

Control how much schema context accompanies the data:

| Level | Form | When |
|-------|------|------|
| L0 | field names only | lightweight models, no `$S` parsing |
| L1 | `$S` + schema ID | capable agents after first contact |
| L2 | `$S` + ID + count + fields | first contact (default) |
| L3 | full schema definition | new consumer, education |
| L4 | natural language fallback | no structured parsing capability |

```python
batch = encoder.encode(records, shadow_level=1)  # abbreviated after first contact
```

## Grouping

`$G` grouping is not included. If you want to group rows by a shared field
(e.g. all chunks from the same source document), do it in your own pipeline
before passing to the prompt builder:

```python
from itertools import groupby

records_by_source = groupby(sorted(records, key=lambda r: r["source"]), key=lambda r: r["source"])
for source, group in records_by_source:
    batch = encoder.encode(list(group), texts=...)
    prompt += f"--- {source} ---\n{batch.to_string()}\n"
```

## Writing Custom Schemas

```json
{
  "$dcp": "schema",
  "id": "your-domain:v1",
  "description": "optional",
  "fields": ["field1", "field2", "field3"],
  "fieldCount": 3,
  "types": {
    "field1": { "type": "string" },
    "field2": { "type": "number", "min": 0, "max": 1 },
    "field3": { "type": ["string", "null"] }
  },
  "origin": {
    "source": "your-api/endpoint",
    "direction": "output"
  }
}
```

`origin` is optional metadata about the data stream:
- `source`: free-form stream identifier (`"tavily/search"`, `"sensor/gyro"`, `"agent/receptor"`)
- `direction`: `"input"` / `"output"` / `"bidirectional"` (default)

## Schema Generator

Infer schema automatically from data samples:

```python
from dcp_py.core.generator import SchemaGenerator

gen = SchemaGenerator()
draft = gen.from_samples(
    samples=[row1, row2, row3],
    domain="my-domain",
    exclude=["internal_id", "embedding_vector"],
)

print(draft.report)       # type inference + enum candidates
draft.save("schemas/my-domain.v1.json")
encoder = draft.to_encoder()
```

## Architecture

```
Layer 0: DcpSchema        ← Schema definition (fields, types, validation)  [universal]
Layer 1: FieldMapping      ← source key → DCP field (dot-notation paths)   [universal]
Layer 2: Preset            ← Per-source defaults (Pinecone, Qdrant, ...)    [domain]
Layer 3: Adapter           ← Per-framework (LlamaIndex, LangChain, ...)     [domain]
```

```
dcp_py/
  core/
    schema.py        ← DcpSchema, SchemaRegistry, FieldType
    mapping.py       ← FieldMapping: dot-notation path resolver
    encoder.py       ← DcpEncoder: $S header, cutdown, DataFrame support
    generator.py     ← SchemaGenerator: infer schema from samples
    controller.py    ← OutputController: place LLM key-value output into DCP
    presets/
      rag/           ← Vector DB presets (pinecone, qdrant, weaviate, chroma, milvus)
  adapters/
    llamaindex.py    ← DcpNodePostprocessor
    langchain.py     ← DcpRunnable
    haystack.py      ← DcpComponent
    azure.py         ← Azure AI Search Custom Skill
  schemas/           ← Built-in DCP schema definitions
```

## Vector DB Presets

| DB | Response structure | Key |
|----|-------------------|-----|
| **Pinecone** | `{ score, metadata: { ... } }` | `DcpEncoder.from_preset("pinecone")` |
| **Qdrant** | `{ score, payload: { ... } }` | `DcpEncoder.from_preset("qdrant")` |
| **Weaviate** | `{ _additional: { score }, properties: { ... } }` | `DcpEncoder.from_preset("weaviate")` |
| **Chroma** | `{ distance, metadata: { ... } }` | `DcpEncoder.from_preset("chroma")` |
| **Milvus** | `{ distance, entity: { ... } }` | `DcpEncoder.from_preset("milvus")` |

## Install

```bash
pip install dcp-py

# With framework extras
pip install "dcp-py[langchain]"
pip install "dcp-py[llamaindex]"
pip install "dcp-py[haystack]"
pip install "dcp-py[azure]"
```

## Related

- [dcp-wrap](https://github.com/hiatamaworkshop/dcp-wrap) — TypeScript equivalent
- [dcp-gateway](https://github.com/hiatamaworkshop/dcp-gateway) — MCP transparent proxy
- [Data Cost Protocol spec](https://dcp-docs.pages.dev)

## License

Apache License 2.0

---

Designed by Hiatama Workshop