## DCP-RAG — Handoff Notes

### What this project is
Data Cost Protocol middleware for RAG pipelines. Drop-in encoder layer that sits between retrieval and LLM generation, converting metadata to native DCP format. Chunk text is never touched — only metadata and inter-stage signals.

### Origin
DCP was designed inside [engram](../engram/) (cross-session memory system). This project extracts the protocol as a standalone RAG middleware. The full DCP spec is in `docs/DATA_COST_PROTOCOL.md`.

### Design decisions made
- **Never touch chunk text** — only metadata (source, score, section) and inter-stage signals (query hints, rerank output)
- **1-line integration** per framework — LlamaIndex `node_postprocessors`, LangChain LCEL pipe, Haystack `@component`
- **Schema-typed positional arrays** — `["docs/auth.md", null, "JWT Config", 0.92, 3]` not `{ source: "docs/auth.md", page: null, ... }`
- **3-layer schema education** — (1) tool/component description embeds schema, (2) errors include schema definition, (3) GET /schemas endpoint for active lookup
- **Backward compatible** — if consumer doesn't support DCP, falls back to original natural language metadata

### Cutdown schema: upstream field loss handling

DCP encoder sits at the LLM boundary — the last layer. Upstream preprocessors (reranker, filter, compressor) may drop or restructure metadata fields before DCP sees them. A template schema defines 5 fields, but actual data may have only 2-3.

**Strategy: detect difference from template → auto-generate cutdown schema header**

```
Template schema: rag-chunk-meta:v1 (5 fields)
  ["$S", "rag-chunk-meta:v1", 5, "source", "page", "section", "score", "chunk_index"]

Actual data: source + score only (page, section, chunk_index missing)

Cutdown: emit reduced $S header with only present fields
  ["$S", "rag-chunk-meta:v1~2", 2, "source", "score"]
  ["docs/auth.md", 0.92]
  ["api/endpoints.yaml", 0.78]
```

#### Field presence bitmask

Each field in the template schema has a fixed bit position (MSB = first field). The encoder scans the batch once, ORing each chunk's presence into a bitmask, then derives the cutdown from the result.

```
rag-chunk-meta:v1 — 5 fields:
  source  page  section  score  chunk_index
  bit4    bit3  bit2     bit1   bit0

Full schema:    0b11111 = 0x1F (31)
source+score:   0b10010 = 0x12 (18)
source+section+score: 0b10110 = 0x16 (22)
```

Cutdown ID encodes the bitmask: `{base_schema}#{hex_mask}`
```
rag-chunk-meta:v1#16  → fields at bits 4,2,1 → source, section, score
```

Detection algorithm (single pass over batch):
```python
mask = 0
for chunk in batch:
    for i, field in enumerate(schema.fields):
        if resolve(chunk, mapping[field]) is not None:
            mask |= (1 << (field_count - 1 - i))

# mask == full_mask → no cutdown needed, use template as-is
# mask != full_mask → emit cutdown $S header with only set-bit fields
```

This is O(chunks × fields) — no allocation, no string comparison, just bitwise OR per field. The `mask == full_mask` check is a single integer comparison to decide whether cutdown is needed.

#### Cutdown output format

```
Template: rag-chunk-meta:v1 (5 fields, full_mask=0x1F)
Batch:    3 chunks, none have "page" or "chunk_index"
Detected: mask=0x16 (0b10110) → source, section, score

Output:
  ["$S", "rag-chunk-meta:v1#16", 3, "source", "section", "score"]
  ["docs/auth.md", "JWT Config", 0.92]
  ["api/users.yaml", "/login", 0.78]
  ["readme.md", "Setup", 0.71]
```

#### Design rules

- **No null-fill** — `[value, null, null, 0.92, null]` wastes tokens, defeats DCP's purpose
- **Cutdown ID = `schema#hex_mask`** — bitmask in the ID itself. Consumer can reconstruct which fields are present without parsing field names. `#` separator distinguishes from `~` (reserved) and `:` (version)
- **$S header includes field names** — even though the mask encodes presence, field names are listed for LLM readability. The mask is for programmatic consumers; field names are for LLM consumers. Both in one header line
- **Schema overhead is negligible** — per DCP spec, schema reference is ID/hash based (`$S` line once per batch, not per chunk). Even with cutdown, overhead is 1 header line regardless of chunk count
- **Batch-level detection** — if a field resolves to null/missing across ALL chunks in the batch, that bit stays 0 and the field is excluded. Per-chunk nulls within otherwise-present fields keep the bit set (e.g. `page` is null for some docs but present for others → bit stays 1, nulls appear in data rows)
- **Full schema preserved upstream** — cutdown only affects DCP output at LLM boundary. Original metadata and template schema remain intact for non-LLM consumers
- **Degenerate case** — if mask == 0 (no fields resolved), encoder skips DCP entirely and falls back to original metadata passthrough
- **No numeric manipulation** — score values are passed as-is (e.g. `0.92`, not `92`). Integer conversion saves trivial tokens but introduces precision risk and data manipulation concerns. DCP encodes structure, never transforms values

### Source grouping: `$G` — chunk structure optimization

Chunk text is never modified. But the **structure** (ordering, delimiting, grouping) is fair game for DCP. When multiple chunks share the same source, repeating source metadata per chunk is redundant. Grouping eliminates this redundancy and gives LLM clearer document-boundary awareness.

#### The problem

```
Typical RAG output to LLM (3 chunks from same doc, 2 from another):

[Result 1]
Source: docs/auth.md
Section: JWT Configuration
Score: 0.92
Content: JWT tokens expire after 24 hours...

[Result 2]
Source: docs/auth.md          ← repeated
Section: OAuth2 Flow
Score: 0.88
Content: The OAuth2 authorization code flow requires...

[Result 3]
Source: docs/auth.md          ← repeated again
Section: Session Management
Score: 0.65
Content: Sessions are stored in Redis with...

[Result 4]
Source: api/endpoints.yaml
Section: /users/login
Score: 0.78
Content: POST /users/login accepts email...

[Result 5]
Source: docs/deploy.md
Section: Docker Setup
Score: 0.71
Content: Run docker compose up -d to start...
```

Source path is repeated per chunk. Delimiter lines (`[Result N]`, `---`) are pure overhead.

#### `$G` group marker

```
["$G", source_value, chunk_count]
```

- `"$G"`: group marker (analogous to `"$S"` for schema)
- `source_value`: the shared source field value
- `chunk_count`: number of chunks in this group (parser knows boundary)

#### Grouped output

```
["$S", "rag-chunk-meta:v1#16", 3, "source", "section", "score"]
["$G", "docs/auth.md", 3]
["JWT Configuration", 0.92]
JWT tokens expire after 24 hours...
["OAuth2 Flow", 0.88]
The OAuth2 authorization code flow requires...
["Session Management", 0.65]
Sessions are stored in Redis with...
["$G", "api/endpoints.yaml", 1]
["/users/login", 0.78]
POST /users/login accepts email...
["$G", "docs/deploy.md", 1]
["Docker Setup", 0.71]
Run docker compose up -d to start...
```

Within a `$G` group, the `source` field is omitted from per-chunk metadata rows (it's in the group header). The cutdown `$S` header reflects the remaining fields.

#### Design rules

- **Group key = `source` field** — the most commonly repeated metadata field in RAG. Other fields are too variable to group on
- **Minimum group size = 1** — even single-chunk sources get a `$G` header for structural consistency. Parser doesn't need conditional logic
- **Sort by group, then by score within group** — chunks from the same source are adjacent. Within a group, highest score first. This is a reorder, not a filter — no data is lost
- **Chunk text is a raw line** — no wrapping, no escaping, no delimiter. The `$S` metadata row before each chunk text serves as the implicit delimiter. Parser alternates: metadata row → text row → metadata row → text row
- **`$G` is optional** — encoder can skip grouping if all sources are unique (no benefit). Detection: count unique sources; if `unique_sources == chunk_count`, skip `$G`
- **Delimiter tokens eliminated** — no `[Result N]`, `---`, `\n\n` separators needed. The structured rows are the delimiters. Saves 3-5 tokens per chunk × chunk_count
- **No data manipulation** — text is passed through verbatim. Order changes within group are the only transformation. DCP does not summarize, truncate, or merge chunk text

#### Token savings estimate

Per chunk, delimiter overhead in typical NL format: ~8-12 tokens (`[Result N]\nSource: ...\n`)
Per chunk, DCP grouped: ~3-4 tokens (metadata array row only, source in group header)
For 10 chunks from 3 sources: ~60-80 tokens saved (delimiter + source dedup)

### Schemas defined (in schemas/)
| Schema | Fields |
|--------|--------|
| `rag-chunk-meta:v1` | source, page, section, score, chunk_index |
| `rag-query-hint:v1` | intent(find\|compare\|summarize\|verify\|expand), domain, detail, urgency |
| `rag-result-summary:v1` | found, count, domain, avg_score |
| `rag-rerank-signal:v1` | chunk_id, original_rank, new_rank, boost_reason |

### Core architecture: 4-layer design

```
Layer 0: DcpSchema      — schema definition (fields, types, validation)
Layer 1: FieldMapping    — metadata key → DCP positional index mapping
Layer 2: Preset          — per-DB default FieldMapping (Pinecone, Qdrant, Weaviate, Chroma, Milvus)
Layer 3: Adapter         — per-framework connector (LlamaIndex, LangChain, Haystack, Azure)
```

Layer 1 (FieldMapping) is the core innovation. Each Vector DB has different response structures:
```
Pinecone:  { id, score, values, metadata: { ... } }
Weaviate:  { _additional: { score, distance }, properties: { ... } }
Qdrant:    { id, score, payload: { ... } }
Chroma:    { ids, distances, metadatas, documents }
Milvus:    { id, distance, entity: { ... } }
```

And metadata inside is user-defined — no universal schema exists.

### Mapping strategy: developer-defined + DB presets

```python
# (1) Preset — 1 line for supported DBs
encoder = DcpEncoder.from_preset("pinecone")

# (2) Preset + overrides — for custom metadata field names
encoder = DcpEncoder.from_preset("qdrant", overrides={
    "section": "payload.heading_text",
})

# (3) Full custom mapping — any DB, any metadata structure
encoder = DcpEncoder(schema="rag-chunk-meta:v1", mapping={
    "source": "metadata.file_path",
    "page": "metadata.page_num",
    "section": "metadata.heading",
    "score": "score",
    "chunk_index": "metadata.idx",
})
```

### Insertion point: LLM input boundary only

reranker, filter, compressor are rule-based or small models — no token cost problem.
DCP encoder sits at the **single point where tokens cost money**: right before the LLM.

```
search → reranker → filter → compressor → [★ DCP encoder ★] → LLM
                                            only here
```

Existing metadata stays intact for upstream stages. DCP is additive, not destructive:
```python
node.metadata = {
    "source": "docs/auth.md",          # ← untouched, upstream stages use this
    "page": 12,
    "_dcp": ["docs/auth.md", 12, "JWT Config", 0.92, 3],
    "_dcp_schema": "rag-chunk-meta:v1"  # ← DCP-aware prompt builder uses this
}
```

#### Scope: input only, no output transformation

DCP-RAG encodes **input to LLM** — it does not transform, intercept, or constrain LLM output. Output format is controlled by the developer's prompt instructions, function calling schema, or response_format settings. DCP has no opinion on what the LLM produces.

#### Multi-LLM pipelines: developer places each encoder

When a pipeline has multiple LLM calls, the developer places a DCP encoder before each one independently. DCP-RAG is not a pipeline orchestrator — it is a single-point encoder.

```
検索 → reranker → [★ DCP encoder A ★] → LLM₁ (query rewrite)
                                           ↓
                       検索₂ → [★ DCP encoder B ★] → LLM₂ (answer generation)
```

- **Each encoder is a separate instance** — different schema, different mapping, different preset if needed
- **Encoder A** might use `rag-query-hint:v1` (signal what to search for)
- **Encoder B** might use `rag-chunk-meta:v1` (annotate retrieved chunks)
- **DCP does not know about the other encoders** — no shared state, no coordination
- **Developer decides where to place them** — DCP provides the tool, not the pipeline design

### What needs to be built
1. **`core/schema.py`** — Schema loader + validator (Layer 0)
2. **`core/mapping.py`** — FieldMapping definition + resolver (Layer 1)
3. **`core/encoder.py`** — DcpEncoder: schema + mapping → native array (ties Layer 0-1)
4. **`core/presets/`** — DB presets: pinecone, qdrant, weaviate, chroma, milvus (Layer 2)
5. **`adapters/llamaindex/`** — `DcpNodePostprocessor` (Layer 3)
6. **`adapters/langchain/`** — `DcpRunnable` (Layer 3)
7. **`adapters/haystack/`** — `DcpComponent` (Layer 3)
8. **`adapters/azure/`** — HTTP Custom Skill endpoint (Layer 3)
9. **Benchmark** — token count: NL metadata vs DCP across real RAG results

### Tech stack decision pending
- Python (LlamaIndex, LangChain, Haystack are all Python-first)
- Core encoder could be language-agnostic (JSON schema + positional array = no language dependency)
- Azure adapter is HTTP, framework-independent

### Key reference from engram
- `engram/gateway/src/schema-registry.ts` — schema loading, validation, field type checking
- `engram/gateway/src/gate/gate.ts` — DCP validator pattern (warn → reject phases)
- `engram/gateway/schemas/` — schema JSON format (`$dcp`, `id`, `fields`, `fieldCount`, `types`, `examples`)
- `engram/docs/DATA_COST_PROTOCOL.md` — full spec including escape hatch, non-recommended patterns, multi-agent handshake

### Target users
Teams running LLM processing pipelines — especially multi-step RAG, agent chains, log analysis, ETL with LLM stages. Any system where AI-to-AI intermediate communication is currently natural language.