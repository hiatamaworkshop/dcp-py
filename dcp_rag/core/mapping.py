"""Layer 1: FieldMapping — metadata key → DCP positional index.

The core innovation of DCP-RAG. Each Vector DB returns results in a different
structure, and user-defined metadata keys vary. FieldMapping bridges the gap
between arbitrary nested metadata and DCP positional arrays.

Supports dot-notation paths for nested access:
  "metadata.file_path"  → result["metadata"]["file_path"]
  "payload.heading"     → result["payload"]["heading"]
  "score"               → result["score"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def resolve_path(obj: Any, path: str) -> Any:
    """Resolve a dot-notation path against a dict-like object.

    Returns None if any segment in the path is missing.

    Examples:
        resolve_path({"a": {"b": 1}}, "a.b") → 1
        resolve_path({"a": {"b": 1}}, "a.c") → None
        resolve_path({"score": 0.9}, "score") → 0.9
    """
    current = obj
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
        if current is None:
            return None
    return current


@dataclass(frozen=True)
class FieldMapping:
    """Mapping from DCP schema fields to source data paths.

    Attributes:
        schema_id: Target DCP schema ID (e.g. "rag-chunk-meta:v1")
        paths: Dict of schema_field_name → dot-notation path in source data
    """

    schema_id: str
    paths: dict[str, str]

    def resolve(self, source: dict[str, Any]) -> dict[str, Any]:
        """Resolve all mapped fields from a source dict.

        Returns dict of schema_field → resolved_value (None if missing).
        """
        return {
            field: resolve_path(source, path)
            for field, path in self.paths.items()
        }

    def resolve_to_row(self, source: dict[str, Any], fields: tuple[str, ...]) -> list[Any]:
        """Resolve fields in schema order, returning a positional array.

        Args:
            source: The source data dict (e.g. a Vector DB result)
            fields: Ordered field names from the schema
        """
        resolved = self.resolve(source)
        return [resolved.get(f) for f in fields]

    def with_overrides(self, overrides: dict[str, str]) -> FieldMapping:
        """Return a new FieldMapping with some paths overridden."""
        new_paths = {**self.paths, **overrides}
        return FieldMapping(schema_id=self.schema_id, paths=new_paths)
