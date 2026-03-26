"""DcpEncoder: schema + mapping → native DCP output.

Ties Layer 0 (schema) and Layer 1 (mapping) together. Handles:
- Bitmask-based cutdown schema detection
- $G source grouping
- $S header generation
- Positional array encoding

This is the main entry point for DCP-RAG.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from dcp_rag.core.schema import DcpSchema, SchemaRegistry, load_default_registry
from dcp_rag.core.mapping import FieldMapping


@dataclass
class EncodedBatch:
    """Result of encoding a batch of chunks.

    Attributes:
        header: $S header as a JSON string
        groups: List of (group_header_or_None, rows) where rows are
                list of (metadata_json, text) tuples
        schema_id: The schema ID used (may be cutdown)
        mask: Field presence bitmask
        is_cutdown: Whether cutdown was applied
        is_grouped: Whether $G grouping was applied
    """

    header: str
    groups: list[tuple[str | None, list[tuple[str, str]]]]
    schema_id: str
    mask: int
    is_cutdown: bool
    is_grouped: bool

    def to_lines(self) -> list[str]:
        """Render the full DCP output as a list of lines."""
        lines = [self.header]
        for group_header, rows in self.groups:
            if group_header is not None:
                lines.append(group_header)
            for meta_json, text in rows:
                lines.append(meta_json)
                lines.append(text)
        return lines

    def to_string(self) -> str:
        """Render the full DCP output as a single string."""
        return "\n".join(self.to_lines())

    def meta_only_lines(self) -> list[str]:
        """Render metadata portion only (no chunk text)."""
        lines = [self.header]
        for group_header, rows in self.groups:
            if group_header is not None:
                lines.append(group_header)
            for meta_json, _text in rows:
                lines.append(meta_json)
        return lines


class DcpEncoder:
    """Main DCP encoder for RAG pipelines.

    Usage:
        # From preset
        encoder = DcpEncoder.from_preset("pinecone")

        # From preset with overrides
        encoder = DcpEncoder.from_preset("qdrant", overrides={"section": "payload.heading_text"})

        # Full custom
        encoder = DcpEncoder(schema="rag-chunk-meta:v1", mapping={
            "source": "metadata.file_path",
            "score": "score",
        })
    """

    def __init__(
        self,
        schema: str | DcpSchema,
        mapping: dict[str, str] | FieldMapping,
        *,
        registry: SchemaRegistry | None = None,
        group_key: str = "source",
        enable_grouping: bool = True,
        text_key: str | None = None,
    ):
        """Initialize encoder.

        Args:
            schema: Schema ID string or DcpSchema instance
            mapping: Dict of {field: dot.path} or FieldMapping instance
            registry: Schema registry. Uses default if None.
            group_key: Schema field to group on (default: "source")
            enable_grouping: Whether to use $G grouping (default: True)
            text_key: Dot-notation path to chunk text in source data.
                      If None, text must be passed separately to encode().
        """
        if registry is None:
            registry = load_default_registry()

        if isinstance(schema, str):
            self._schema = registry.get(schema)
        else:
            self._schema = schema

        if isinstance(mapping, dict):
            self._mapping = FieldMapping(
                schema_id=self._schema.id, paths=mapping
            )
        else:
            self._mapping = mapping

        self._group_key = group_key
        self._enable_grouping = enable_grouping
        self._text_key = text_key

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        *,
        schema: str = "rag-chunk-meta:v1",
        overrides: dict[str, str] | None = None,
        registry: SchemaRegistry | None = None,
        **kwargs: Any,
    ) -> DcpEncoder:
        """Create encoder from a DB preset.

        Args:
            preset_name: Preset name (e.g. "pinecone", "qdrant")
            schema: Schema ID (default: rag-chunk-meta:v1)
            overrides: Override specific field paths
            registry: Schema registry
            **kwargs: Passed to DcpEncoder.__init__
        """
        from dcp_rag.core.presets import get_preset

        mapping = get_preset(preset_name, schema_id=schema)
        if overrides:
            mapping = mapping.with_overrides(overrides)
        return cls(schema=schema, mapping=mapping, registry=registry, **kwargs)

    def detect_mask(self, resolved_batch: list[dict[str, Any]]) -> int:
        """Detect field presence bitmask across a batch.

        Single pass: O(chunks × fields), bitwise OR per field.
        """
        mask = 0
        fc = self._schema.field_count
        for resolved in resolved_batch:
            for i, fname in enumerate(self._schema.fields):
                if resolved.get(fname) is not None:
                    mask |= 1 << (fc - 1 - i)
        return mask

    def _should_group(self, resolved_batch: list[dict[str, Any]]) -> bool:
        """Determine if $G grouping would help."""
        if not self._enable_grouping:
            return False
        if self._group_key not in self._schema.fields:
            return False
        sources = {r.get(self._group_key) for r in resolved_batch}
        # Skip grouping if all sources are unique (no benefit)
        return len(sources) < len(resolved_batch)

    def _group_batch(
        self, resolved_batch: list[dict[str, Any]], texts: list[str]
    ) -> OrderedDict[Any, list[tuple[dict[str, Any], str]]]:
        """Group resolved chunks by group_key, sorted by score within group."""
        groups: OrderedDict[Any, list[tuple[dict[str, Any], str]]] = OrderedDict()
        for resolved, text in zip(resolved_batch, texts):
            key = resolved.get(self._group_key, None)
            if key not in groups:
                groups[key] = []
            groups[key].append((resolved, text))

        # Sort within each group by score descending
        for key in groups:
            groups[key].sort(
                key=lambda pair: pair[0].get("score") or 0, reverse=True
            )
        return groups

    def encode(
        self,
        chunks: list[dict[str, Any]],
        texts: list[str] | None = None,
        *,
        shadow_level: int = 2,
    ) -> EncodedBatch:
        """Encode a batch of chunks into DCP format.

        Args:
            chunks: List of source data dicts (e.g. Vector DB results)
            texts: List of chunk texts. If None, extracted via text_key.
            shadow_level: Header density level (0-4).
                0 = fields only, 1 = with schema ID, 2 = full protocol (default),
                3 = full schema def, 4 = NL fallback.

        Returns:
            EncodedBatch with header, optional $G groups, and data rows.
            For shadow_level=4, data rows are NL key-value strings instead of arrays.
        """
        if not chunks:
            return EncodedBatch(
                header="", groups=[], schema_id=self._schema.id,
                mask=0, is_cutdown=False, is_grouped=False,
            )

        # Extract texts
        if texts is None:
            if self._text_key is None:
                raise ValueError(
                    "Either pass texts= or set text_key in constructor"
                )
            from dcp_rag.core.mapping import resolve_path
            texts = [resolve_path(c, self._text_key) or "" for c in chunks]

        if len(texts) != len(chunks):
            raise ValueError(
                f"chunks ({len(chunks)}) and texts ({len(texts)}) length mismatch"
            )

        # Resolve all mappings
        resolved_batch = [self._mapping.resolve(c) for c in chunks]

        # Detect cutdown
        mask = self.detect_mask(resolved_batch)
        full_mask = self._schema.full_mask
        is_cutdown = mask != full_mask

        # Degenerate case: no fields resolved → skip DCP
        if mask == 0:
            return EncodedBatch(
                header="", groups=[], schema_id=self._schema.id,
                mask=0, is_cutdown=False, is_grouped=False,
            )

        # Build header at requested shadow level
        active_fields = self._schema.fields_from_mask(mask)
        schema_id = self._schema.cutdown_id(mask)
        header_obj = self._schema.s_header_at_level(mask, shadow_level=shadow_level)
        header = json.dumps(header_obj) if not isinstance(header_obj, str) else header_obj

        # NL fallback (L4): key-value text instead of positional arrays
        if shadow_level >= 4:
            rows = []
            for resolved, text in zip(resolved_batch, texts):
                parts = [
                    f"{f}: {resolved.get(f)}" for f in active_fields
                    if resolved.get(f) is not None
                ]
                rows.append((", ".join(parts), text))
            return EncodedBatch(
                header=header, groups=[(None, rows)], schema_id=schema_id,
                mask=mask, is_cutdown=is_cutdown, is_grouped=False,
            )

        # Decide grouping (L0-L3: positional arrays)
        use_grouping = self._should_group(resolved_batch)

        if use_grouping:
            grouped = self._group_batch(resolved_batch, texts)
            # Fields for per-row data (group_key goes in $G header)
            row_fields = tuple(f for f in active_fields if f != self._group_key)

            groups = []
            for source_val, pairs in grouped.items():
                g_header = json.dumps(["$G", source_val, len(pairs)])
                rows = []
                for resolved, text in pairs:
                    row = [resolved.get(f) for f in row_fields]
                    rows.append((json.dumps(row), text))
                groups.append((g_header, rows))

            return EncodedBatch(
                header=header, groups=groups, schema_id=schema_id,
                mask=mask, is_cutdown=is_cutdown, is_grouped=True,
            )
        else:
            # No grouping — flat list
            rows = []
            for resolved, text in zip(resolved_batch, texts):
                row = [resolved.get(f) for f in active_fields]
                rows.append((json.dumps(row), text))

            return EncodedBatch(
                header=header, groups=[(None, rows)], schema_id=schema_id,
                mask=mask, is_cutdown=is_cutdown, is_grouped=False,
            )

    def encode_metadata(
        self,
        chunk: dict[str, Any],
    ) -> dict[str, Any]:
        """Encode a single chunk's metadata as _dcp fields.

        Returns a dict with _dcp (positional array) and _dcp_schema keys,
        suitable for merging into the chunk's metadata dict.

        This is for per-node metadata injection (e.g. LlamaIndex postprocessor).
        """
        resolved = self._mapping.resolve(chunk)
        mask = 0
        fc = self._schema.field_count
        for i, fname in enumerate(self._schema.fields):
            if resolved.get(fname) is not None:
                mask |= 1 << (fc - 1 - i)

        if mask == 0:
            return {}

        active_fields = self._schema.fields_from_mask(mask)
        row = [resolved.get(f) for f in active_fields]

        return {
            "_dcp": row,
            "_dcp_schema": self._schema.cutdown_id(mask),
        }
