"""Azure AI Search Custom Skill adapter: HTTP endpoint.

Usage:
    uvicorn dcp_py.adapters.azure:app --host 0.0.0.0 --port 8080

Azure AI Search Custom Skills expect:
  POST / with { values: [{ recordId, data: { ... } }] }
  Response: { values: [{ recordId, data: { _dcp, _dcp_schema }, errors: [], warnings: [] }] }
"""

from __future__ import annotations

import os
from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "fastapi is required for Azure adapter. "
        "Install with: pip install dcp-rag[azure]"
    )

from dcp_py.core.encoder import DcpEncoder


# -- Request/Response models --

class SkillInputRecord(BaseModel):
    recordId: str
    data: dict[str, Any]


class SkillInput(BaseModel):
    values: list[SkillInputRecord]


class SkillOutputRecord(BaseModel):
    recordId: str
    data: dict[str, Any]
    errors: list[str]
    warnings: list[str]


class SkillOutput(BaseModel):
    values: list[SkillOutputRecord]


# -- App --

app = FastAPI(title="DCP-RAG Azure Custom Skill")

# Encoder configured via environment variables
_encoder: DcpEncoder | None = None


def _get_encoder() -> DcpEncoder:
    global _encoder
    if _encoder is None:
        preset = os.environ.get("DCP_PRESET", "pinecone")
        schema = os.environ.get("DCP_SCHEMA", "rag-chunk-meta:v1")
        _encoder = DcpEncoder.from_preset(preset, schema=schema)
    return _encoder


@app.post("/", response_model=SkillOutput)
async def process_skill(request: SkillInput) -> SkillOutput:
    encoder = _get_encoder()
    results = []
    for record in request.values:
        errors = []
        warnings = []
        try:
            dcp_fields = encoder.encode_metadata(record.data)
        except Exception as e:
            dcp_fields = {}
            errors.append(str(e))

        results.append(SkillOutputRecord(
            recordId=record.recordId,
            data=dcp_fields,
            errors=errors,
            warnings=warnings,
        ))
    return SkillOutput(values=results)


@app.get("/schemas")
async def list_schemas() -> dict[str, Any]:
    from dcp_py.core.schema import load_default_registry
    registry = load_default_registry()
    return {"schemas": registry.list()}
