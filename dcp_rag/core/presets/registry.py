"""DB preset registry — predefined FieldMappings for common Vector DBs.

Each preset defines default dot-notation paths for the rag-chunk-meta:v1 schema
based on the DB's native response structure.
"""

from __future__ import annotations

from dcp_rag.core.mapping import FieldMapping


# -- Preset definitions --
# Each maps rag-chunk-meta:v1 fields to the DB's response structure.
# Score fields point to the DB's native score/distance field.
# Metadata fields use the DB's metadata container key as prefix.

_PRESETS: dict[str, dict[str, dict[str, str]]] = {
    "pinecone": {
        # { id, score, values, metadata: { source, page, section, chunk_index } }
        "rag-chunk-meta:v1": {
            "source": "metadata.source",
            "page": "metadata.page",
            "section": "metadata.section",
            "score": "score",
            "chunk_index": "metadata.chunk_index",
        },
    },
    "qdrant": {
        # { id, score, payload: { source, page, section, chunk_index } }
        "rag-chunk-meta:v1": {
            "source": "payload.source",
            "page": "payload.page",
            "section": "payload.section",
            "score": "score",
            "chunk_index": "payload.chunk_index",
        },
    },
    "weaviate": {
        # { _additional: { score, distance }, properties: { source, page, section, chunk_index } }
        "rag-chunk-meta:v1": {
            "source": "properties.source",
            "page": "properties.page",
            "section": "properties.section",
            "score": "_additional.score",
            "chunk_index": "properties.chunk_index",
        },
    },
    "chroma": {
        # Chroma returns batch format: { ids[], distances[], metadatas[], documents[] }
        # Pre-processing required to flatten into per-chunk dicts.
        # This preset assumes already-flattened per-chunk dicts:
        # { id, distance, metadata: { source, page, section, chunk_index } }
        "rag-chunk-meta:v1": {
            "source": "metadata.source",
            "page": "metadata.page",
            "section": "metadata.section",
            "score": "distance",
            "chunk_index": "metadata.chunk_index",
        },
    },
    "milvus": {
        # { id, distance, entity: { source, page, section, chunk_index } }
        "rag-chunk-meta:v1": {
            "source": "entity.source",
            "page": "entity.page",
            "section": "entity.section",
            "score": "distance",
            "chunk_index": "entity.chunk_index",
        },
    },
}


def get_preset(
    db_name: str,
    schema_id: str = "rag-chunk-meta:v1",
) -> FieldMapping:
    """Get a preset FieldMapping for a Vector DB.

    Args:
        db_name: DB name (pinecone, qdrant, weaviate, chroma, milvus)
        schema_id: Target schema ID

    Raises:
        KeyError: If preset or schema not found
    """
    db_name = db_name.lower()
    if db_name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise KeyError(
            f"no preset for {db_name!r}. Available: [{available}]"
        )
    schemas = _PRESETS[db_name]
    if schema_id not in schemas:
        available = ", ".join(sorted(schemas.keys()))
        raise KeyError(
            f"preset {db_name!r} has no mapping for schema {schema_id!r}. "
            f"Available: [{available}]"
        )
    return FieldMapping(schema_id=schema_id, paths=schemas[schema_id])


def list_presets() -> dict[str, list[str]]:
    """Return dict of db_name → [supported schema IDs]."""
    return {db: sorted(schemas.keys()) for db, schemas in _PRESETS.items()}
