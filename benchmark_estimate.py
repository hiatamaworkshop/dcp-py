"""
DCP-RAG token cost estimation.

Compares:
  A) Natural language metadata (typical RAG prompt injection)
  B) JSON key-value metadata
  C) DCP full schema (positional array + $S header)
  D) DCP cutdown schema (bitmask, null fields removed)

Scenarios:
  1) Single-stage RAG: 5 chunks, chunk-meta only
  2) Multi-stage RAG: 10 chunks, chunk-meta + query-hint + result-summary + rerank-signal
  3) Scaled: 20 chunks, all schemas, some fields missing (cutdown kicks in)
"""

import json

# ---------- token estimation ----------
# GPT-4 / Claude average: ~1 token per 4 chars for English mixed content
# JSON/code is slightly worse (~1 token per 3.5 chars) due to punctuation
def estimate_tokens(text: str, ratio: float = 3.8) -> int:
    return max(1, round(len(text) / ratio))


# ---------- sample data ----------
CHUNKS_5 = [
    {"source": "docs/auth.md", "page": None, "section": "JWT Configuration", "score": 0.92, "chunk_index": 3,
     "text": "JWT tokens expire after 24 hours. Refresh tokens are valid for 30 days..."},
    {"source": "docs/auth.md", "page": None, "section": "OAuth2 Flow", "score": 0.88, "chunk_index": 4,
     "text": "The OAuth2 authorization code flow requires a client_id and redirect_uri..."},
    {"source": "contracts/2024-Q1.pdf", "page": 12, "section": "Payment Terms", "score": 0.87, "chunk_index": 0,
     "text": "Payment is due within 30 days of invoice date. Late payments incur 1.5% monthly..."},
    {"source": "api/endpoints.yaml", "page": None, "section": "/users/login", "score": 0.78, "chunk_index": 5,
     "text": "POST /users/login accepts email and password, returns JWT access_token..."},
    {"source": "docs/deploy.md", "page": None, "section": "Docker Setup", "score": 0.71, "chunk_index": 1,
     "text": "Run docker compose up -d to start all services. The API binds to port 8080..."},
]

CHUNKS_10 = CHUNKS_5 + [
    {"source": "docs/api-reference.md", "page": None, "section": "Rate Limiting", "score": 0.69, "chunk_index": 2,
     "text": "Rate limit is 100 requests per minute per API key. Returns 429 on exceed..."},
    {"source": "docs/security.md", "page": 3, "section": "CORS Policy", "score": 0.67, "chunk_index": 0,
     "text": "CORS is configured to allow origins listed in ALLOWED_ORIGINS env var..."},
    {"source": "docs/auth.md", "page": None, "section": "Session Management", "score": 0.65, "chunk_index": 7,
     "text": "Sessions are stored in Redis with a 4-hour TTL. Session rotation on privilege..."},
    {"source": "runbooks/incident-2024-02.md", "page": 1, "section": "Root Cause", "score": 0.62, "chunk_index": 0,
     "text": "The outage was caused by a connection pool exhaustion under sustained load..."},
    {"source": "docs/monitoring.md", "page": None, "section": "Health Checks", "score": 0.58, "chunk_index": 3,
     "text": "GET /health returns 200 with component status. Checks DB, Redis, and queue..."},
]

# Chunks with missing fields (simulating upstream preprocessor dropping fields)
CHUNKS_20_SPARSE = CHUNKS_10 + [
    {"source": "docs/billing.md", "score": 0.55, "text": "Billing cycles reset on the 1st of each month..."},
    {"source": "docs/billing.md", "score": 0.53, "text": "Overage charges apply at $0.01 per 1000 tokens..."},
    {"source": "api/webhooks.yaml", "score": 0.51, "text": "Webhooks are sent via POST with HMAC-SHA256 signature..."},
    {"source": "docs/migration.md", "score": 0.49, "text": "Run alembic upgrade head to apply pending migrations..."},
    {"source": "docs/testing.md", "score": 0.47, "text": "Integration tests require TEST_DB_URL to be set..."},
    {"source": "docs/caching.md", "score": 0.45, "text": "Response caching uses ETag headers with 5-minute TTL..."},
    {"source": "docs/logging.md", "score": 0.43, "text": "Structured logs use JSON format with correlation IDs..."},
    {"source": "docs/rbac.md", "score": 0.41, "text": "Roles are admin, editor, viewer. Permissions are additive..."},
    {"source": "docs/i18n.md", "score": 0.39, "text": "Translations are loaded from locale/*.json at startup..."},
    {"source": "docs/pagination.md", "score": 0.37, "text": "Cursor-based pagination using encoded (id, timestamp) pairs..."},
]

QUERY_HINT = {"intent": "find", "domain": "auth", "detail": "JWT expiration policy", "urgency": 0.9}
RESULT_SUMMARY = {"found": True, "count": 5, "domain": "auth", "avg_score": 0.83}
RERANK_SIGNALS = [
    {"chunk_id": "chunk_0", "original_rank": 3, "new_rank": 1, "boost_reason": "exact_match"},
    {"chunk_id": "chunk_1", "original_rank": 1, "new_rank": 2, "boost_reason": None},
    {"chunk_id": "chunk_2", "original_rank": 5, "new_rank": 3, "boost_reason": "recency"},
]

SCHEMA_FIELDS = {
    "rag-chunk-meta:v1": ["source", "page", "section", "score", "chunk_index"],
    "rag-query-hint:v1": ["intent", "domain", "detail", "urgency"],
    "rag-result-summary:v1": ["found", "count", "domain", "avg_score"],
    "rag-rerank-signal:v1": ["chunk_id", "original_rank", "new_rank", "boost_reason"],
}


# ---------- Format A: Natural Language ----------
def format_nl_chunk(chunk: dict, rank: int) -> str:
    parts = [f"[Result {rank}]"]
    parts.append(f"Source: {chunk['source']}")
    if chunk.get("page") is not None:
        parts.append(f"Page: {chunk['page']}")
    if chunk.get("section") is not None:
        parts.append(f"Section: {chunk['section']}")
    parts.append(f"Relevance Score: {chunk['score']}")
    if chunk.get("chunk_index") is not None:
        parts.append(f"Chunk Index: {chunk['chunk_index']}")
    parts.append(f"Content: {chunk['text']}")
    return "\n".join(parts)

def format_nl_query_hint(hint: dict) -> str:
    return (f"Query Intent: {hint['intent']}\n"
            f"Domain: {hint['domain']}\n"
            f"Detail: {hint['detail']}\n"
            f"Urgency: {hint['urgency']}")

def format_nl_result_summary(summary: dict) -> str:
    return (f"Results Found: {'Yes' if summary['found'] else 'No'}\n"
            f"Result Count: {summary['count']}\n"
            f"Primary Domain: {summary['domain']}\n"
            f"Average Score: {summary['avg_score']}")

def format_nl_rerank(signals: list) -> str:
    lines = []
    for s in signals:
        reason = s['boost_reason'] or 'none'
        lines.append(f"Chunk {s['chunk_id']}: rank {s['original_rank']} → {s['new_rank']} (reason: {reason})")
    return "\n".join(lines)


# ---------- Format B: JSON key-value ----------
def format_json_chunk(chunk: dict) -> str:
    meta = {k: v for k, v in chunk.items() if k != "text" and v is not None}
    return json.dumps({"metadata": meta, "content": chunk["text"]}, ensure_ascii=False)

def format_json_signals(hint, summary, reranks) -> str:
    return json.dumps({"query_hint": hint, "result_summary": summary, "rerank_signals": reranks}, ensure_ascii=False)


# ---------- Format C: DCP full schema ----------
def format_dcp_header(schema_id: str) -> str:
    fields = SCHEMA_FIELDS[schema_id]
    return json.dumps(["$S", schema_id, len(fields)] + fields)

def format_dcp_chunk(chunk: dict, fields: list) -> str:
    return json.dumps([chunk.get(f) for f in fields])

def format_dcp_signal(data: dict, fields: list) -> str:
    return json.dumps([data.get(f) for f in fields])


# ---------- Format E: DCP with $G grouping ----------
def group_chunks(chunks: list) -> dict:
    """Group chunks by source, preserving order of first appearance."""
    from collections import OrderedDict
    groups = OrderedDict()
    for c in chunks:
        src = c.get("source", "unknown")
        if src not in groups:
            groups[src] = []
        groups[src].append(c)
    # Sort within group by score descending
    for src in groups:
        groups[src].sort(key=lambda x: x.get("score", 0), reverse=True)
    return groups

def format_dcp_grouped(chunks: list, schema_id: str, fields: list, use_cutdown: bool = False) -> str:
    """Full DCP output with $G grouping. Returns (meta_str, content_str)."""
    groups = group_chunks(chunks)

    # Detect cutdown
    mask = detect_mask(chunks, fields)
    full_mask = (1 << len(fields)) - 1

    if use_cutdown and mask != full_mask:
        active = cutdown_fields(fields, mask)
        # Remove 'source' from per-row fields (it's in $G header)
        row_fields = [f for f in active if f != "source"]
        header = json.dumps(["$S", f"{schema_id}#{mask:x}", len(active)] + active)
    else:
        row_fields = [f for f in fields if f != "source"]
        header = json.dumps(["$S", schema_id, len(fields)] + fields)

    lines = [header]
    for src, group in groups.items():
        lines.append(json.dumps(["$G", src, len(group)]))
        for c in group:
            lines.append(json.dumps([c.get(f) for f in row_fields]))
            lines.append(c["text"])

    return "\n".join(lines)

def format_dcp_grouped_meta_only(chunks: list, schema_id: str, fields: list, use_cutdown: bool = False) -> str:
    """Meta portion only (no chunk text) for token comparison."""
    groups = group_chunks(chunks)

    mask = detect_mask(chunks, fields)
    full_mask = (1 << len(fields)) - 1

    if use_cutdown and mask != full_mask:
        active = cutdown_fields(fields, mask)
        row_fields = [f for f in active if f != "source"]
        header = json.dumps(["$S", f"{schema_id}#{mask:x}", len(active)] + active)
    else:
        row_fields = [f for f in fields if f != "source"]
        header = json.dumps(["$S", schema_id, len(fields)] + fields)

    lines = [header]
    for src, group in groups.items():
        lines.append(json.dumps(["$G", src, len(group)]))
        for c in group:
            lines.append(json.dumps([c.get(f) for f in row_fields]))

    return "\n".join(lines)


# ---------- Format D: DCP cutdown ----------
def detect_mask(chunks: list, fields: list) -> int:
    mask = 0
    fc = len(fields)
    for chunk in chunks:
        for i, f in enumerate(fields):
            if chunk.get(f) is not None:
                mask |= (1 << (fc - 1 - i))
    return mask

def cutdown_fields(fields: list, mask: int) -> list:
    fc = len(fields)
    return [f for i, f in enumerate(fields) if mask & (1 << (fc - 1 - i))]

def format_dcp_cutdown_header(schema_id: str, mask: int) -> str:
    fields = SCHEMA_FIELDS[schema_id]
    active = cutdown_fields(fields, mask)
    cut_id = f"{schema_id}#{mask:x}"
    return json.dumps(["$S", cut_id, len(active)] + active)

def format_dcp_cutdown_chunk(chunk: dict, fields: list, mask: int) -> str:
    active = cutdown_fields(fields, mask)
    return json.dumps([chunk.get(f) for f in active])


# ---------- Scenarios ----------
def scenario_1():
    """Single-stage RAG: 5 chunks, chunk-meta only"""
    print("=" * 70)
    print("SCENARIO 1: Single-stage RAG — 5 chunks, chunk-meta only")
    print("=" * 70)

    fields = SCHEMA_FIELDS["rag-chunk-meta:v1"]

    # A) Natural language
    nl = "\n\n".join(format_nl_chunk(c, i+1) for i, c in enumerate(CHUNKS_5))
    nl_tokens = estimate_tokens(nl)

    # Separate: metadata vs content
    nl_meta_only = "\n\n".join(
        "\n".join(format_nl_chunk(c, i+1).split("\n")[:-1])
        for i, c in enumerate(CHUNKS_5)
    )
    nl_content_only = "\n\n".join(c["text"] for c in CHUNKS_5)
    nl_meta_tokens = estimate_tokens(nl_meta_only)
    nl_content_tokens = estimate_tokens(nl_content_only)

    # B) JSON
    js = "\n".join(format_json_chunk(c) for c in CHUNKS_5)
    js_tokens = estimate_tokens(js)

    # C) DCP full
    dcp_header = format_dcp_header("rag-chunk-meta:v1")
    dcp_rows = "\n".join(format_dcp_chunk(c, fields) for c in CHUNKS_5)
    dcp_content = "\n".join(c["text"] for c in CHUNKS_5)
    dcp_full = dcp_header + "\n" + dcp_rows + "\n" + dcp_content
    dcp_meta = dcp_header + "\n" + dcp_rows
    dcp_tokens = estimate_tokens(dcp_full)
    dcp_meta_tokens = estimate_tokens(dcp_meta)

    # D) DCP cutdown (same as full here — all fields present in 5 chunks)
    mask = detect_mask(CHUNKS_5, fields)
    full_mask = (1 << len(fields)) - 1
    cutdown_note = "same as full (all fields present)" if mask == full_mask else f"mask=0x{mask:x}"

    print(f"\n{'Format':<25} {'Total tokens':>14} {'Meta tokens':>14} {'Content tokens':>14} {'Meta ratio':>12}")
    print("-" * 80)
    print(f"{'A) Natural language':<25} {nl_tokens:>14} {nl_meta_tokens:>14} {nl_content_tokens:>14} {'baseline':>12}")
    print(f"{'B) JSON key-value':<25} {js_tokens:>14} {'—':>14} {'—':>14} {'—':>12}")
    print(f"{'C) DCP full schema':<25} {dcp_tokens:>14} {dcp_meta_tokens:>14} {nl_content_tokens:>14} {dcp_meta_tokens/nl_meta_tokens*100:>10.0f}%")
    print(f"{'D) DCP cutdown':<25} {'(' + cutdown_note + ')':>50}")

    print(f"\nMeta-only savings (NL → DCP): {nl_meta_tokens - dcp_meta_tokens} tokens ({(1 - dcp_meta_tokens/nl_meta_tokens)*100:.0f}% reduction)")
    print(f"Content (unchanged): {nl_content_tokens} tokens")
    print(f"Total savings: {nl_tokens - dcp_tokens} tokens ({(1 - dcp_tokens/nl_tokens)*100:.0f}% of total prompt)")

    return nl_meta_tokens, dcp_meta_tokens


def scenario_2():
    """Multi-stage RAG: 10 chunks, all signal types"""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Multi-stage RAG — 10 chunks + inter-stage signals")
    print("=" * 70)

    fields = SCHEMA_FIELDS["rag-chunk-meta:v1"]

    # A) Natural language
    nl_chunks = "\n\n".join(format_nl_chunk(c, i+1) for i, c in enumerate(CHUNKS_10))
    nl_signals = "\n\n".join([
        format_nl_query_hint(QUERY_HINT),
        format_nl_result_summary(RESULT_SUMMARY),
        format_nl_rerank(RERANK_SIGNALS),
    ])
    nl_full = nl_chunks + "\n\n" + nl_signals
    nl_meta = "\n\n".join(
        "\n".join(format_nl_chunk(c, i+1).split("\n")[:-1])
        for i, c in enumerate(CHUNKS_10)
    ) + "\n\n" + nl_signals
    nl_content = "\n\n".join(c["text"] for c in CHUNKS_10)
    nl_meta_tokens = estimate_tokens(nl_meta)
    nl_content_tokens = estimate_tokens(nl_content)
    nl_total_tokens = estimate_tokens(nl_full)

    # C) DCP full
    dcp_parts = []
    # chunk meta
    dcp_parts.append(format_dcp_header("rag-chunk-meta:v1"))
    for c in CHUNKS_10:
        dcp_parts.append(format_dcp_chunk(c, fields))
    # query hint
    dcp_parts.append(format_dcp_header("rag-query-hint:v1"))
    dcp_parts.append(format_dcp_signal(QUERY_HINT, SCHEMA_FIELDS["rag-query-hint:v1"]))
    # result summary
    dcp_parts.append(format_dcp_header("rag-result-summary:v1"))
    dcp_parts.append(format_dcp_signal(RESULT_SUMMARY, SCHEMA_FIELDS["rag-result-summary:v1"]))
    # rerank
    dcp_parts.append(format_dcp_header("rag-rerank-signal:v1"))
    for r in RERANK_SIGNALS:
        dcp_parts.append(format_dcp_signal(r, SCHEMA_FIELDS["rag-rerank-signal:v1"]))
    dcp_meta = "\n".join(dcp_parts)
    dcp_meta_tokens = estimate_tokens(dcp_meta)

    dcp_full = dcp_meta + "\n" + nl_content
    dcp_total_tokens = estimate_tokens(dcp_full)

    print(f"\n{'Format':<25} {'Total tokens':>14} {'Meta tokens':>14} {'Content tokens':>14} {'Meta ratio':>12}")
    print("-" * 80)
    print(f"{'A) Natural language':<25} {nl_total_tokens:>14} {nl_meta_tokens:>14} {nl_content_tokens:>14} {'baseline':>12}")
    print(f"{'C) DCP full schema':<25} {dcp_total_tokens:>14} {dcp_meta_tokens:>14} {nl_content_tokens:>14} {dcp_meta_tokens/nl_meta_tokens*100:>10.0f}%")

    print(f"\nMeta-only savings: {nl_meta_tokens - dcp_meta_tokens} tokens ({(1 - dcp_meta_tokens/nl_meta_tokens)*100:.0f}% reduction)")
    print(f"Total savings: {nl_total_tokens - dcp_total_tokens} tokens ({(1 - dcp_total_tokens/nl_total_tokens)*100:.0f}% of total prompt)")


def scenario_3():
    """Scaled: 20 chunks, sparse fields → cutdown kicks in"""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Scaled RAG — 20 chunks, sparse metadata (cutdown active)")
    print("=" * 70)

    fields = SCHEMA_FIELDS["rag-chunk-meta:v1"]
    full_mask = (1 << len(fields)) - 1

    # Normalize: ensure all chunks have all keys (missing → None)
    chunks = []
    for c in CHUNKS_20_SPARSE:
        normalized = {f: c.get(f) for f in fields}
        normalized["text"] = c["text"]
        chunks.append(normalized)

    mask = detect_mask(chunks, fields)
    active_fields = cutdown_fields(fields, mask)

    # A) Natural language
    nl_meta = "\n\n".join(
        "\n".join(format_nl_chunk(c, i+1).split("\n")[:-1])
        for i, c in enumerate(chunks)
    )
    nl_content = "\n\n".join(c["text"] for c in chunks)
    nl_meta_tokens = estimate_tokens(nl_meta)
    nl_content_tokens = estimate_tokens(nl_content)

    # C) DCP full (with nulls)
    dcp_full_header = format_dcp_header("rag-chunk-meta:v1")
    dcp_full_rows = "\n".join(format_dcp_chunk(c, fields) for c in chunks)
    dcp_full_meta = dcp_full_header + "\n" + dcp_full_rows
    dcp_full_meta_tokens = estimate_tokens(dcp_full_meta)

    # D) DCP cutdown
    dcp_cut_header = format_dcp_cutdown_header("rag-chunk-meta:v1", mask)
    dcp_cut_rows = "\n".join(format_dcp_cutdown_chunk(c, fields, mask) for c in chunks)
    dcp_cut_meta = dcp_cut_header + "\n" + dcp_cut_rows
    dcp_cut_meta_tokens = estimate_tokens(dcp_cut_meta)

    print(f"\nField presence mask: 0b{mask:05b} = 0x{mask:x}")
    print(f"Template fields: {fields}")
    print(f"Active fields:   {active_fields}")
    print(f"Dropped fields:  {[f for f in fields if f not in active_fields]}")

    print(f"\n{'Format':<25} {'Meta tokens':>14} {'Content tokens':>14} {'Meta ratio':>12}")
    print("-" * 65)
    print(f"{'A) Natural language':<25} {nl_meta_tokens:>14} {nl_content_tokens:>14} {'baseline':>12}")
    print(f"{'C) DCP full (nulls)':<25} {dcp_full_meta_tokens:>14} {nl_content_tokens:>14} {dcp_full_meta_tokens/nl_meta_tokens*100:>10.0f}%")
    print(f"{'D) DCP cutdown':<25} {dcp_cut_meta_tokens:>14} {nl_content_tokens:>14} {dcp_cut_meta_tokens/nl_meta_tokens*100:>10.0f}%")

    print(f"\nDCP full savings:    {nl_meta_tokens - dcp_full_meta_tokens} tokens ({(1 - dcp_full_meta_tokens/nl_meta_tokens)*100:.0f}%)")
    print(f"DCP cutdown savings: {nl_meta_tokens - dcp_cut_meta_tokens} tokens ({(1 - dcp_cut_meta_tokens/nl_meta_tokens)*100:.0f}%)")
    print(f"Cutdown vs full:     {dcp_full_meta_tokens - dcp_cut_meta_tokens} tokens saved by removing null fields")


def scenario_3b():
    """Pure sparse: all 20 chunks have only source + score (preprocessor stripped everything else)"""
    print("\n" + "=" * 70)
    print("SCENARIO 3b: Sparse-only — 20 chunks, preprocessor dropped page/section/chunk_index")
    print("=" * 70)

    fields = SCHEMA_FIELDS["rag-chunk-meta:v1"]
    full_mask = (1 << len(fields)) - 1

    # All chunks have only source + score
    chunks = []
    all_sources = [c["source"] for c in CHUNKS_20_SPARSE]
    all_scores = [c.get("score", 0.5) for c in CHUNKS_20_SPARSE]
    all_texts = [c["text"] for c in CHUNKS_20_SPARSE]
    for i in range(20):
        chunks.append({
            "source": all_sources[i], "page": None, "section": None,
            "score": all_scores[i], "chunk_index": None,
            "text": all_texts[i],
        })

    mask = detect_mask(chunks, fields)
    active_fields = cutdown_fields(fields, mask)

    # A) Natural language
    nl_meta = "\n\n".join(
        "\n".join(format_nl_chunk(c, i+1).split("\n")[:-1])
        for i, c in enumerate(chunks)
    )
    nl_content = "\n\n".join(c["text"] for c in chunks)
    nl_meta_tokens = estimate_tokens(nl_meta)
    nl_content_tokens = estimate_tokens(nl_content)

    # C) DCP full (with nulls)
    dcp_full_header = format_dcp_header("rag-chunk-meta:v1")
    dcp_full_rows = "\n".join(format_dcp_chunk(c, fields) for c in chunks)
    dcp_full_meta = dcp_full_header + "\n" + dcp_full_rows
    dcp_full_meta_tokens = estimate_tokens(dcp_full_meta)

    # D) DCP cutdown
    dcp_cut_header = format_dcp_cutdown_header("rag-chunk-meta:v1", mask)
    dcp_cut_rows = "\n".join(format_dcp_cutdown_chunk(c, fields, mask) for c in chunks)
    dcp_cut_meta = dcp_cut_header + "\n" + dcp_cut_rows
    dcp_cut_meta_tokens = estimate_tokens(dcp_cut_meta)

    print(f"\nField presence mask: 0b{mask:05b} = 0x{mask:x}")
    print(f"Template fields: {fields}")
    print(f"Active fields:   {active_fields}")
    print(f"Dropped fields:  {[f for f in fields if f not in active_fields]}")

    print(f"\n{'Format':<25} {'Meta tokens':>14} {'Content tokens':>14} {'Meta ratio':>12}")
    print("-" * 65)
    print(f"{'A) Natural language':<25} {nl_meta_tokens:>14} {nl_content_tokens:>14} {'baseline':>12}")
    print(f"{'C) DCP full (nulls)':<25} {dcp_full_meta_tokens:>14} {nl_content_tokens:>14} {dcp_full_meta_tokens/nl_meta_tokens*100:>10.0f}%")
    print(f"{'D) DCP cutdown':<25} {dcp_cut_meta_tokens:>14} {nl_content_tokens:>14} {dcp_cut_meta_tokens/nl_meta_tokens*100:>10.0f}%")

    print(f"\nNL → DCP full:      {nl_meta_tokens - dcp_full_meta_tokens} tokens saved ({(1 - dcp_full_meta_tokens/nl_meta_tokens)*100:.0f}%)")
    print(f"NL → DCP cutdown:   {nl_meta_tokens - dcp_cut_meta_tokens} tokens saved ({(1 - dcp_cut_meta_tokens/nl_meta_tokens)*100:.0f}%)")
    print(f"Full → Cutdown:     {dcp_full_meta_tokens - dcp_cut_meta_tokens} tokens saved by removing null fields")
    print(f"Cutdown vs Full:    {(1 - dcp_cut_meta_tokens/dcp_full_meta_tokens)*100:.0f}% further reduction")


def scenario_4():
    """$G grouping: 10 chunks from 3 sources (realistic RAG distribution)"""
    print("\n" + "=" * 70)
    print("SCENARIO 4: $G grouping — 10 chunks from 3 sources")
    print("=" * 70)

    # Realistic: auth doc dominates (4 chunks), api doc (3), deploy doc (3)
    grouped_chunks = [
        {"source": "docs/auth.md", "page": None, "section": "JWT Configuration", "score": 0.92, "chunk_index": 3,
         "text": "JWT tokens expire after 24 hours. Refresh tokens are valid for 30 days..."},
        {"source": "docs/auth.md", "page": None, "section": "OAuth2 Flow", "score": 0.88, "chunk_index": 4,
         "text": "The OAuth2 authorization code flow requires a client_id and redirect_uri..."},
        {"source": "docs/auth.md", "page": None, "section": "Session Management", "score": 0.65, "chunk_index": 7,
         "text": "Sessions are stored in Redis with a 4-hour TTL. Session rotation on privilege..."},
        {"source": "docs/auth.md", "page": None, "section": "RBAC", "score": 0.61, "chunk_index": 9,
         "text": "Role-based access uses three tiers: admin, editor, viewer. Permissions are additive..."},
        {"source": "api/endpoints.yaml", "page": None, "section": "/users/login", "score": 0.78, "chunk_index": 5,
         "text": "POST /users/login accepts email and password, returns JWT access_token..."},
        {"source": "api/endpoints.yaml", "page": None, "section": "/users/refresh", "score": 0.72, "chunk_index": 6,
         "text": "POST /users/refresh accepts refresh_token, returns new access_token..."},
        {"source": "api/endpoints.yaml", "page": None, "section": "/users/logout", "score": 0.58, "chunk_index": 8,
         "text": "POST /users/logout invalidates the current session and refresh token..."},
        {"source": "docs/deploy.md", "page": None, "section": "Docker Setup", "score": 0.71, "chunk_index": 1,
         "text": "Run docker compose up -d to start all services. The API binds to port 8080..."},
        {"source": "docs/deploy.md", "page": None, "section": "Environment Variables", "score": 0.67, "chunk_index": 2,
         "text": "Required env vars: DATABASE_URL, REDIS_URL, JWT_SECRET, CORS_ORIGINS..."},
        {"source": "docs/deploy.md", "page": None, "section": "Health Checks", "score": 0.55, "chunk_index": 10,
         "text": "GET /health returns 200 with component status. Checks DB, Redis, and queue..."},
    ]

    fields = SCHEMA_FIELDS["rag-chunk-meta:v1"]
    unique_sources = len(set(c["source"] for c in grouped_chunks))

    # A) Natural language — full prompt
    nl_full = "\n\n".join(format_nl_chunk(c, i+1) for i, c in enumerate(grouped_chunks))
    nl_tokens = estimate_tokens(nl_full)

    nl_meta = "\n\n".join(
        "\n".join(format_nl_chunk(c, i+1).split("\n")[:-1])
        for i, c in enumerate(grouped_chunks)
    )
    nl_content = "\n\n".join(c["text"] for c in grouped_chunks)
    nl_meta_tokens = estimate_tokens(nl_meta)
    nl_content_tokens = estimate_tokens(nl_content)

    # C) DCP without grouping
    dcp_header = format_dcp_header("rag-chunk-meta:v1")
    dcp_rows = "\n".join(format_dcp_chunk(c, fields) for c in grouped_chunks)
    dcp_meta = dcp_header + "\n" + dcp_rows
    dcp_meta_tokens = estimate_tokens(dcp_meta)

    # E) DCP with $G grouping
    dcp_g_meta = format_dcp_grouped_meta_only(grouped_chunks, "rag-chunk-meta:v1", fields)
    dcp_g_meta_tokens = estimate_tokens(dcp_g_meta)
    dcp_g_full = format_dcp_grouped(grouped_chunks, "rag-chunk-meta:v1", fields)
    dcp_g_full_tokens = estimate_tokens(dcp_g_full)

    print(f"\n10 chunks from {unique_sources} sources: docs/auth.md(4), api/endpoints.yaml(3), docs/deploy.md(3)")

    print(f"\n{'Format':<30} {'Total':>10} {'Meta':>10} {'Content':>10} {'vs NL meta':>12} {'vs NL total':>12}")
    print("-" * 85)
    print(f"{'A) Natural language':<30} {nl_tokens:>10} {nl_meta_tokens:>10} {nl_content_tokens:>10} {'baseline':>12} {'baseline':>12}")
    print(f"{'C) DCP (no grouping)':<30} {dcp_meta_tokens + nl_content_tokens:>10} {dcp_meta_tokens:>10} {nl_content_tokens:>10} {dcp_meta_tokens/nl_meta_tokens*100:>10.0f}% {(dcp_meta_tokens + nl_content_tokens)/nl_tokens*100:>10.0f}%")
    print(f"{'E) DCP + $G grouping':<30} {dcp_g_full_tokens:>10} {dcp_g_meta_tokens:>10} {nl_content_tokens:>10} {dcp_g_meta_tokens/nl_meta_tokens*100:>10.0f}% {dcp_g_full_tokens/nl_tokens*100:>10.0f}%")

    print(f"\n--- Savings breakdown ---")
    print(f"NL → DCP (no group):  meta {nl_meta_tokens - dcp_meta_tokens:>4} tokens saved ({(1 - dcp_meta_tokens/nl_meta_tokens)*100:.0f}%)")
    print(f"NL → DCP + $G:        meta {nl_meta_tokens - dcp_g_meta_tokens:>4} tokens saved ({(1 - dcp_g_meta_tokens/nl_meta_tokens)*100:.0f}%)")
    print(f"$G incremental gain:       {dcp_meta_tokens - dcp_g_meta_tokens:>4} tokens ({(1 - dcp_g_meta_tokens/dcp_meta_tokens)*100:.0f}% further reduction from grouping)")
    print(f"\nTotal prompt reduction: {nl_tokens - dcp_g_full_tokens} tokens ({(1 - dcp_g_full_tokens/nl_tokens)*100:.0f}% of total NL prompt)")


def scenario_5():
    """Worst case for grouping: 10 chunks, all unique sources"""
    print("\n" + "=" * 70)
    print("SCENARIO 5: $G worst case — 10 chunks, all unique sources")
    print("=" * 70)

    fields = SCHEMA_FIELDS["rag-chunk-meta:v1"]

    # A) NL
    nl_meta = "\n\n".join(
        "\n".join(format_nl_chunk(c, i+1).split("\n")[:-1])
        for i, c in enumerate(CHUNKS_10)
    )
    nl_content = "\n\n".join(c["text"] for c in CHUNKS_10)
    nl_meta_tokens = estimate_tokens(nl_meta)
    nl_content_tokens = estimate_tokens(nl_content)
    nl_tokens = estimate_tokens("\n\n".join(format_nl_chunk(c, i+1) for i, c in enumerate(CHUNKS_10)))

    # C) DCP no grouping
    dcp_header = format_dcp_header("rag-chunk-meta:v1")
    dcp_rows = "\n".join(format_dcp_chunk(c, fields) for c in CHUNKS_10)
    dcp_meta = dcp_header + "\n" + dcp_rows
    dcp_meta_tokens = estimate_tokens(dcp_meta)

    # E) DCP with $G (all unique = 10 groups of 1)
    dcp_g_meta = format_dcp_grouped_meta_only(CHUNKS_10, "rag-chunk-meta:v1", fields)
    dcp_g_meta_tokens = estimate_tokens(dcp_g_meta)

    print(f"\nAll 10 chunks from different sources (no grouping benefit)")

    print(f"\n{'Format':<30} {'Meta tokens':>14} {'vs NL meta':>12}")
    print("-" * 60)
    print(f"{'A) Natural language':<30} {nl_meta_tokens:>14} {'baseline':>12}")
    print(f"{'C) DCP (no grouping)':<30} {dcp_meta_tokens:>14} {dcp_meta_tokens/nl_meta_tokens*100:>10.0f}%")
    print(f"{'E) DCP + $G':<30} {dcp_g_meta_tokens:>14} {dcp_g_meta_tokens/nl_meta_tokens*100:>10.0f}%")

    overhead = dcp_g_meta_tokens - dcp_meta_tokens
    print(f"\n$G overhead (all unique): {overhead:+d} tokens ({'cost' if overhead > 0 else 'neutral'})")
    print(f"(Each $G header for a single-chunk group adds ~6 tokens but removes source from data row)")


def cost_projection():
    """Real-world cost projection"""
    print("\n" + "=" * 70)
    print("COST PROJECTION: Production RAG workload")
    print("=" * 70)

    # Assumptions
    queries_per_day = 10_000
    chunks_per_query = 10
    avg_meta_tokens_nl = 45  # per chunk, NL metadata
    avg_meta_tokens_dcp = 15  # per chunk, DCP
    schema_overhead = 20  # $S header, once per query
    price_per_1m_input = 3.00  # $/1M tokens (Claude Sonnet tier)

    daily_meta_nl = queries_per_day * chunks_per_query * avg_meta_tokens_nl
    daily_meta_dcp = queries_per_day * (chunks_per_query * avg_meta_tokens_dcp + schema_overhead)
    daily_savings = daily_meta_nl - daily_meta_dcp
    monthly_savings_usd = (daily_savings * 30 / 1_000_000) * price_per_1m_input

    print(f"\nAssumptions:")
    print(f"  Queries/day:           {queries_per_day:,}")
    print(f"  Chunks/query:          {chunks_per_query}")
    print(f"  NL meta tokens/chunk:  ~{avg_meta_tokens_nl}")
    print(f"  DCP meta tokens/chunk: ~{avg_meta_tokens_dcp}")
    print(f"  Price:                 ${price_per_1m_input}/1M input tokens")

    print(f"\nDaily metadata tokens:")
    print(f"  NL:  {daily_meta_nl:>12,}")
    print(f"  DCP: {daily_meta_dcp:>12,}")
    print(f"  Δ:   {daily_savings:>12,} ({daily_savings/daily_meta_nl*100:.0f}% reduction)")

    print(f"\nMonthly cost savings: ${monthly_savings_usd:,.2f}")
    print(f"Annual cost savings:  ${monthly_savings_usd * 12:,.2f}")

    print(f"\nNote: This is metadata cost only. Chunk content tokens are")
    print(f"unchanged. In a typical RAG prompt, metadata is 15-30% of total")
    print(f"input tokens. DCP reduces that portion by ~65-75%.")


if __name__ == "__main__":
    scenario_1()
    scenario_2()
    scenario_3()
    scenario_3b()
    scenario_4()
    scenario_5()
    cost_projection()
