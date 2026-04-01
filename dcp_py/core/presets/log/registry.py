"""Log preset registry — predefined FieldMappings for common log sources.

Each preset defines default dot-notation paths for the log-entry:v1 schema
based on the source's native log structure.
"""

from __future__ import annotations

from dcp_py.core.mapping import FieldMapping


# -- Preset definitions --
# Each maps log-entry:v1 fields to the source's native log structure.
# Loki preset assumes pre-parsed dicts (raw Loki format requires custom flattening).

_PRESETS: dict[str, dict[str, dict[str, str]]] = {
    "cloudwatch": {
        # CloudWatch Logs Insights result fields
        # { timestamp, level, logStreamName, message, ... }
        "log-entry:v1": {
            "ts":      "timestamp",
            "level":   "level",
            "service": "logStreamName",
            "msg":     "message",
        },
    },
    "datadog": {
        # Datadog log record fields
        # { timestamp, status, service, message, ... }
        "log-entry:v1": {
            "ts":      "timestamp",
            "level":   "status",
            "service": "service",
            "msg":     "message",
        },
    },
    "loki": {
        # Pre-parsed Loki log dicts (raw format is complex: { values[], labels{} })
        # This preset assumes already-flattened per-entry dicts:
        # { ts, level, labels: { app, ... }, line }
        "log-entry:v1": {
            "ts":      "ts",
            "level":   "level",
            "service": "labels.app",
            "msg":     "line",
        },
    },
    "generic": {
        # Generic structured log format — flat dict with standard field names
        # { ts, level, service, msg }
        "log-entry:v1": {
            "ts":      "ts",
            "level":   "level",
            "service": "service",
            "msg":     "msg",
        },
    },
}


def get_log_preset(
    source_name: str,
    schema_id: str = "log-entry:v1",
) -> FieldMapping:
    """Get a preset FieldMapping for a log source.

    Args:
        source_name: Log source name (cloudwatch, datadog, loki, generic)
        schema_id: Target schema ID

    Raises:
        KeyError: If preset or schema not found
    """
    source_name = source_name.lower()
    if source_name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise KeyError(
            f"no log preset for {source_name!r}. Available: [{available}]"
        )
    schemas = _PRESETS[source_name]
    if schema_id not in schemas:
        available = ", ".join(sorted(schemas.keys()))
        raise KeyError(
            f"preset {source_name!r} has no mapping for schema {schema_id!r}. "
            f"Available: [{available}]"
        )
    return FieldMapping(schema_id=schema_id, paths=schemas[source_name][schema_id])


def list_log_presets() -> dict[str, list[str]]:
    """Return dict of source_name → [supported schema IDs]."""
    return {source: sorted(schemas.keys()) for source, schemas in _PRESETS.items()}