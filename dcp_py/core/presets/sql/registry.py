"""SQL preset registry — predefined FieldMappings for common SQL sources.

Each preset defines direct field paths for the sql-row-meta:v1 schema.
All supported sources use flat dict rows (direct field mapping), so all
paths are identity mappings.

For encoding actual data columns (not metadata), use DcpEncoder.from_dataframe()
instead — it auto-infers schema fields from the DataFrame columns at runtime.
"""

from __future__ import annotations

from dcp_py.core.mapping import FieldMapping


# -- Preset definitions --
# All SQL sources expose row metadata as flat dicts, so paths are direct
# field names with no dot-notation nesting required.

_PRESETS: dict[str, dict[str, dict[str, str]]] = {
    "psycopg2": {
        # dict rows from psycopg2 RealDictCursor — already flat dicts
        "sql-row-meta:v1": {
            "db":       "db",
            "table":    "table",
            "row_num":  "row_num",
            "query_ms": "query_ms",
        },
    },
    "sqlalchemy": {
        # SQLAlchemy Row converted to dict via ._mapping or dict(row)
        "sql-row-meta:v1": {
            "db":       "db",
            "table":    "table",
            "row_num":  "row_num",
            "query_ms": "query_ms",
        },
    },
    "sqlite3": {
        # sqlite3.Row converted to dict via dict(row)
        "sql-row-meta:v1": {
            "db":       "db",
            "table":    "table",
            "row_num":  "row_num",
            "query_ms": "query_ms",
        },
    },
    "generic": {
        # Generic dict — any SQL source that produces flat row dicts
        "sql-row-meta:v1": {
            "db":       "db",
            "table":    "table",
            "row_num":  "row_num",
            "query_ms": "query_ms",
        },
    },
}


def get_sql_preset(
    source_name: str,
    schema_id: str = "sql-row-meta:v1",
) -> FieldMapping:
    """Get a preset FieldMapping for a SQL source.

    Args:
        source_name: SQL source name (psycopg2, sqlalchemy, sqlite3, generic)
        schema_id: Target schema ID

    Raises:
        KeyError: If preset or schema not found
    """
    source_name = source_name.lower()
    if source_name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise KeyError(
            f"no sql preset for {source_name!r}. Available: [{available}]"
        )
    schemas = _PRESETS[source_name]
    if schema_id not in schemas:
        available = ", ".join(sorted(schemas.keys()))
        raise KeyError(
            f"sql preset {source_name!r} has no mapping for schema {schema_id!r}. "
            f"Available: [{available}]"
        )
    return FieldMapping(schema_id=schema_id, paths=schemas[schema_id])


def list_sql_presets() -> dict[str, list[str]]:
    """Return dict of source_name → [supported schema IDs]."""
    return {source: sorted(schemas.keys()) for source, schemas in _PRESETS.items()}