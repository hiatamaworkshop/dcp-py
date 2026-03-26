"""Schema Generator — infer DcpSchema + FieldMapping from data samples.

Takes raw data samples (dicts) and produces a DCP schema definition with
type inference, enum detection, field ordering, and auto-generated mapping.

The generator embeds DCP conventions so that any caller (human or AI)
gets a compliant schema without knowing the rules.

Usage:
    gen = SchemaGenerator()
    draft = gen.from_samples(
        samples=[{"score": 0.9, "source": "docs/auth.md", ...}, ...],
        domain="rag-chunk-meta",
    )
    schema = draft.schema      # DcpSchema
    mapping = draft.mapping    # FieldMapping
    draft.save("schemas/rag-chunk-meta.v1.json")
    encoder = draft.to_encoder()
"""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

from dcp_rag.core.schema import DcpSchema, FieldType, SchemaRegistry
from dcp_rag.core.mapping import FieldMapping
from dcp_rag.core.encoder import DcpEncoder


# ── Field ordering heuristics ────────────────────────────────
# DCP convention: identifiers first, then classifiers, numerics, text.
# Within each category, higher-frequency fields come first (survive cutdown).

_CATEGORY_ORDER = {
    "identifier": 0,   # source, id, name, path, endpoint
    "classifier": 1,   # status, level, type, action, method
    "numeric": 2,       # score, count, weight, latency, page
    "text": 3,          # summary, detail, description, section
    "other": 4,
}

_IDENTIFIER_HINTS = frozenset({
    "id", "source", "name", "path", "endpoint", "url", "uri", "key",
    "file", "file_path", "doc", "document", "chunk_id", "node_id",
})

_CLASSIFIER_HINTS = frozenset({
    "status", "level", "type", "action", "method", "kind", "category",
    "state", "trigger", "mode", "role", "domain",
})

_NUMERIC_HINTS = frozenset({
    "score", "count", "weight", "latency", "page", "rank", "index",
    "chunk_index", "position", "size", "duration", "confidence",
    "distance", "similarity", "uptime", "hit_count",
})


def _classify_field(name: str, values: list[Any]) -> str:
    """Classify a field into a category for ordering."""
    lower = name.lower()
    if lower in _IDENTIFIER_HINTS:
        return "identifier"
    if lower in _CLASSIFIER_HINTS:
        return "classifier"
    if lower in _NUMERIC_HINTS:
        return "numeric"
    # Infer from values
    non_null = [v for v in values if v is not None]
    if non_null and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return "numeric"
    if non_null and all(isinstance(v, str) and len(v) > 50 for v in non_null):
        return "text"
    if non_null and all(isinstance(v, str) for v in non_null):
        unique_ratio = len(set(non_null)) / len(non_null) if non_null else 1
        if unique_ratio < 0.3:
            return "classifier"
    return "other"


def _infer_type(values: list[Any]) -> dict[str, Any]:
    """Infer FieldType attributes from observed values."""
    non_null = [v for v in values if v is not None]
    has_null = len(non_null) < len(values)

    if not non_null:
        types = ["null"]
        return {"type": types}

    # Detect types present
    type_set: set[str] = set()
    for v in non_null:
        if isinstance(v, bool):
            type_set.add("boolean")
        elif isinstance(v, (int, float)):
            type_set.add("number")
        elif isinstance(v, str):
            type_set.add("string")
        else:
            type_set.add("string")  # fallback

    types: list[str] = sorted(type_set)
    if has_null:
        types.append("null")

    result: dict[str, Any] = {}
    result["type"] = types[0] if len(types) == 1 else types

    # Enum detection: few unique string values relative to sample count
    if "string" in type_set and len(type_set) == 1:
        unique_vals = sorted(set(non_null))
        if 2 <= len(unique_vals) <= 10 and len(unique_vals) <= len(non_null) * 0.6:
            result["enum"] = unique_vals

    # Numeric range detection
    if "number" in type_set and len(type_set) == 1:
        nums = [v for v in non_null if isinstance(v, (int, float))]
        if nums:
            lo, hi = min(nums), max(nums)
            # Only set bounds if they suggest a constrained range
            if lo >= 0 and hi <= 1:
                result["min"] = 0
                result["max"] = 1
            elif lo >= 0:
                result["min"] = 0

    return result


def _flatten_keys(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict into dot-notation keys with leaf values."""
    result: dict[str, Any] = {}
    for k, v in obj.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_keys(v, full_key))
        else:
            result[full_key] = v
    return result


# ── SchemaDraft ──────────────────────────────────────────────

@dataclass
class FieldReport:
    """Inference report for a single field."""
    name: str
    source_path: str
    category: str
    inferred_type: dict[str, Any]
    presence_rate: float
    unique_count: int
    sample_count: int
    is_group_key_candidate: bool = False


@dataclass
class SchemaDraft:
    """Result of schema generation. Can be inspected, adjusted, and saved."""

    schema: DcpSchema
    mapping: FieldMapping
    field_reports: list[FieldReport] = dc_field(default_factory=list)

    @property
    def report(self) -> str:
        """Human-readable inference report."""
        lines = [f"Schema: {self.schema.id}", f"Fields: {len(self.schema.fields)}", ""]
        for fr in self.field_reports:
            t = fr.inferred_type.get("type", "?")
            enum = fr.inferred_type.get("enum")
            flags = []
            if fr.is_group_key_candidate:
                flags.append("group_key candidate")
            if enum:
                flags.append(f"enum({len(enum)})")
            if fr.presence_rate < 1.0:
                flags.append(f"nullable({fr.presence_rate:.0%})")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            lines.append(
                f"  {fr.name}: {t} "
                f"(source: {fr.source_path}, "
                f"unique: {fr.unique_count}/{fr.sample_count})"
                f"{flag_str}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export schema as JSON-serializable dict."""
        types_dict = {}
        for fname in self.schema.fields:
            ft = self.schema.types.get(fname)
            if ft:
                td: dict[str, Any] = {"type": ft.type}
                if ft.description:
                    td["description"] = ft.description
                if ft.enum is not None:
                    td["enum"] = ft.enum
                if ft.min is not None:
                    td["min"] = ft.min
                if ft.max is not None:
                    td["max"] = ft.max
                types_dict[fname] = td

        return {
            "$dcp": "schema",
            "id": self.schema.id,
            "description": self.schema.description,
            "fields": list(self.schema.fields),
            "fieldCount": self.schema.field_count,
            "types": types_dict,
        }

    def save(self, path: str | Path) -> None:
        """Save schema definition to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")

    def to_encoder(self, **kwargs: Any) -> DcpEncoder:
        """Create a DcpEncoder from this draft."""
        return DcpEncoder(schema=self.schema, mapping=self.mapping, **kwargs)


# ── SchemaGenerator ──────────────────────────────────────────

class SchemaGenerator:
    """Infer DcpSchema + FieldMapping from data samples.

    Embeds DCP conventions:
    - Field ordering: identifiers → classifiers → numerics → text
    - Enum detection for low-cardinality string fields
    - Numeric range detection (0-1 scores, non-negative counts)
    - Group key candidate detection (high-repetition fields)
    - Auto-mapping via dot-notation path resolution
    """

    def from_samples(
        self,
        samples: list[dict[str, Any]],
        domain: str,
        *,
        version: int = 1,
        description: str = "",
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        field_names: dict[str, str] | None = None,
    ) -> SchemaDraft:
        """Generate a schema draft from data samples.

        Args:
            samples: List of source data dicts (e.g. DB results, API responses)
            domain: Domain name for schema ID (e.g. "rag-chunk-meta")
            version: Schema version number (default: 1)
            description: Schema description
            include: If set, only include these dot-notation paths
            exclude: Paths to exclude (e.g. internal IDs, embeddings)
            field_names: Override mapping from source path to schema field name
                         e.g. {"metadata.file_path": "source"}

        Returns:
            SchemaDraft with inferred schema, mapping, and report
        """
        if not samples:
            raise ValueError("need at least 1 sample")

        exclude_set = set(exclude or [])
        field_names = field_names or {}

        # Step 1: Flatten all samples and collect per-path values
        path_values: OrderedDict[str, list[Any]] = OrderedDict()
        for sample in samples:
            flat = _flatten_keys(sample)
            seen_paths = set()
            for path, value in flat.items():
                if path in exclude_set:
                    continue
                if include and path not in include:
                    continue
                if path not in path_values:
                    path_values[path] = []
                path_values[path].append(value)
                seen_paths.add(path)
            # Mark missing paths as None
            for path in path_values:
                if path not in seen_paths:
                    path_values[path].append(None)

        if not path_values:
            raise ValueError("no fields found in samples after filtering")

        # Step 2: Analyze each field
        analyzed: list[tuple[str, str, str, dict[str, Any], list[Any]]] = []
        # (schema_field_name, source_path, category, type_info, values)

        for source_path, values in path_values.items():
            schema_name = field_names.get(source_path, source_path.split(".")[-1])
            category = _classify_field(schema_name, values)
            type_info = _infer_type(values)
            analyzed.append((schema_name, source_path, category, type_info, values))

        # Step 3: Sort by DCP convention
        # Primary: category order. Secondary: presence rate (descending).
        def sort_key(item: tuple) -> tuple:
            name, path, cat, tinfo, vals = item
            non_null = sum(1 for v in vals if v is not None)
            presence = non_null / len(vals) if vals else 0
            return (_CATEGORY_ORDER.get(cat, 99), -presence, name)

        analyzed.sort(key=sort_key)

        # Step 4: Deduplicate field names (different paths → same leaf name)
        seen_names: dict[str, int] = {}
        final_fields: list[tuple[str, str, str, dict[str, Any], list[Any]]] = []
        for schema_name, source_path, category, type_info, values in analyzed:
            if schema_name in seen_names:
                seen_names[schema_name] += 1
                schema_name = f"{schema_name}_{seen_names[schema_name]}"
            else:
                seen_names[schema_name] = 0
            final_fields.append((schema_name, source_path, category, type_info, values))

        # Step 5: Build schema and mapping
        schema_id = f"{domain}:v{version}"
        field_order = tuple(f[0] for f in final_fields)

        types_dict: dict[str, FieldType] = {}
        mapping_paths: dict[str, str] = {}
        reports: list[FieldReport] = []

        for schema_name, source_path, category, type_info, values in final_fields:
            # Build FieldType
            types_dict[schema_name] = FieldType(
                type=type_info.get("type", "string"),
                enum=type_info.get("enum"),
                min=type_info.get("min"),
                max=type_info.get("max"),
            )

            # Build mapping (auto-bind: if leaf name == schema name and no nesting, skip path)
            mapping_paths[schema_name] = source_path

            # Report
            non_null = [v for v in values if v is not None]
            unique_vals = set(str(v) for v in non_null)
            presence_rate = len(non_null) / len(values) if values else 0
            repetition_rate = 1 - (len(unique_vals) / len(non_null)) if non_null else 0

            reports.append(FieldReport(
                name=schema_name,
                source_path=source_path,
                category=category,
                inferred_type=type_info,
                presence_rate=presence_rate,
                unique_count=len(unique_vals),
                sample_count=len(values),
                is_group_key_candidate=repetition_rate > 0.3 and category in ("identifier", "classifier"),
            ))

        schema = DcpSchema(
            id=schema_id,
            description=description,
            fields=field_order,
            field_count=len(field_order),
            types=types_dict,
        )

        mapping = FieldMapping(
            schema_id=schema_id,
            paths=mapping_paths,
        )

        return SchemaDraft(
            schema=schema,
            mapping=mapping,
            field_reports=reports,
        )