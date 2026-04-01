"""DcpEncoder: schema + mapping → native DCP output.

Ties Layer 0 (schema) and Layer 1 (mapping) together. Handles:
- Bitmask-based cutdown schema detection
- $S header generation at configurable shadow levels
- Positional array encoding with '-' for absent values

Note on $G grouping: not included in the encoder. If you want to group
rows by a shared field (e.g. source document), do it in your own pipeline
before passing to the prompt. The encoder's job is encoding, not layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dcp_py.core.schema import DcpSchema, SchemaRegistry, load_default_registry
from dcp_py.core.mapping import FieldMapping

_ABSENT = "-"


@dataclass
class EncodedBatch:
    """Result of encoding a batch of records.

    Attributes:
        header: $S header as a JSON string (empty if no fields resolved)
        rows: List of (row_json, text) tuples. text is "" when no text source.
        schema_id: The schema ID used (may be cutdown)
        mask: Field presence bitmask
        is_cutdown: Whether cutdown was applied
    """

    header: str
    rows: list[tuple[str, str]]
    schema_id: str
    mask: int
    is_cutdown: bool

    def to_lines(self) -> list[str]:
        """Render the full DCP output as a list of lines.

        If text is present, each record is two lines: row JSON + text.
        If text is empty, each record is one line: row JSON only.
        """
        lines = [self.header]
        for row_json, text in self.rows:
            lines.append(row_json)
            if text:
                lines.append(text)
        return lines

    def to_string(self) -> str:
        """Render the full DCP output as a single string."""
        return "\n".join(self.to_lines())

    def meta_only_lines(self) -> list[str]:
        """Render metadata rows only (no text)."""
        return [self.header] + [row_json for row_json, _ in self.rows]


class DcpEncoder:
    """Universal DCP encoder: dict → positional array.

    Works for any structured data going into an LLM context window:
    RAG chunks, SQL rows, log entries, API responses, sensor readings, etc.

    Usage:
        # From preset (RAG / Vector DB)
        encoder = DcpEncoder.from_preset("pinecone")

        # From preset with field path overrides
        encoder = DcpEncoder.from_preset("qdrant", overrides={"section": "payload.heading_text"})

        # Full custom (any domain)
        encoder = DcpEncoder(schema="rag-chunk-meta:v1", mapping={
            "source": "metadata.file_path",
            "score":  "score",
        })

        # From a pandas DataFrame
        encoder, batch = DcpEncoder.from_dataframe(df, domain="query-result")
    """

    def __init__(
        self,
        schema: str | DcpSchema,
        mapping: dict[str, str] | FieldMapping,
        *,
        registry: SchemaRegistry | None = None,
        text_key: str | None = None,
    ):
        """Initialize encoder.

        Args:
            schema: Schema ID string or DcpSchema instance
            mapping: Dict of {field: dot.path} or FieldMapping instance
            registry: Schema registry. Uses default if None.
            text_key: Dot-notation path to a text body field in source data.
                      If set, encode() extracts text automatically.
                      If None, pass texts= explicitly or omit for metadata-only.
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
        """Create encoder from a Vector DB / RAG preset.

        Args:
            preset_name: Preset name (e.g. "pinecone", "qdrant")
            schema: Schema ID (default: rag-chunk-meta:v1)
            overrides: Override specific field paths
            registry: Schema registry
            **kwargs: Passed to DcpEncoder.__init__
        """
        from dcp_py.core.presets import get_preset

        mapping = get_preset(preset_name, schema_id=schema)
        if overrides:
            mapping = mapping.with_overrides(overrides)
        return cls(schema=schema, mapping=mapping, registry=registry, **kwargs)

    @classmethod
    def from_dataframe(
        cls,
        df: Any,  # pandas DataFrame
        *,
        domain: str = "query-result",
        version: int = 1,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        shadow_level: int = 2,
    ) -> tuple[DcpEncoder, EncodedBatch]:
        """Encode a pandas DataFrame directly to DCP.

        Infers schema from column names and dtypes. Returns both the
        encoder (reusable for subsequent batches) and the encoded batch.

        Args:
            df: pandas DataFrame
            domain: Schema domain name → schema ID "domain:vN"
            version: Schema version number
            include: Column names to include (default: all)
            exclude: Column names to exclude
            shadow_level: Header density (0-4, default 2)

        Returns:
            (DcpEncoder, EncodedBatch)

        Example:
            encoder, batch = DcpEncoder.from_dataframe(df, domain="sales-q1")
            print(batch.to_string())
        """
        from dcp_py.core.generator import SchemaGenerator

        cols = list(df.columns)
        if include:
            cols = [c for c in cols if c in include]
        if exclude:
            cols = [c for c in cols if c not in exclude]

        records = df[cols].to_dict(orient="records")

        gen = SchemaGenerator()
        draft = gen.from_samples(
            samples=records,
            domain=domain,
            version=version,
            include=cols,
        )
        encoder = cls(schema=draft.schema, mapping=draft.mapping)
        batch = encoder.encode(records, shadow_level=shadow_level)
        return encoder, batch

    def detect_mask(self, resolved_batch: list[dict[str, Any]]) -> int:
        """Detect field presence bitmask across a batch.

        A field bit is set if any record has a non-None value for that field.
        Single pass: O(records × fields).
        """
        mask = 0
        fc = self._schema.field_count
        for resolved in resolved_batch:
            for i, fname in enumerate(self._schema.fields):
                if resolved.get(fname) is not None:
                    mask |= 1 << (fc - 1 - i)
        return mask

    def encode(
        self,
        records: list[dict[str, Any]],
        texts: list[str] | None = None,
        *,
        shadow_level: int = 2,
    ) -> EncodedBatch:
        """Encode a batch of records into DCP format.

        Args:
            records: List of source data dicts (Vector DB results, SQL rows, etc.)
            texts: Optional list of text bodies (one per record). If None and
                   text_key is set, extracted automatically. Pass [] or omit
                   for metadata-only encoding.
            shadow_level: Header density level (0-4).
                0 = field names only (no $S, no schema ID)
                1 = $S + schema ID
                2 = $S + ID + field count + field names (default)
                3 = full schema definition as JSON object
                4 = natural language fallback (key: value per field)

        Returns:
            EncodedBatch. Use .to_string() for the final context window payload.

        Note on absent values: fields with None are represented as '-' in
        the positional array. Fields with no data across the entire batch
        are omitted via bitmask cutdown.
        """
        if not records:
            return EncodedBatch(
                header="", rows=[], schema_id=self._schema.id,
                mask=0, is_cutdown=False,
            )

        # Extract texts
        if texts is None:
            if self._text_key is not None:
                from dcp_py.core.mapping import resolve_path
                texts = [resolve_path(c, self._text_key) or "" for c in records]
            else:
                texts = [""] * len(records)

        if len(texts) != len(records):
            raise ValueError(
                f"records ({len(records)}) and texts ({len(texts)}) length mismatch"
            )

        # Resolve all mappings
        resolved_batch = [self._mapping.resolve(c) for c in records]

        # Detect cutdown
        mask = self.detect_mask(resolved_batch)
        is_cutdown = mask != self._schema.full_mask

        # Degenerate: no fields resolved
        if mask == 0:
            return EncodedBatch(
                header="", rows=[], schema_id=self._schema.id,
                mask=0, is_cutdown=False,
            )

        active_fields = self._schema.fields_from_mask(mask)
        schema_id = self._schema.cutdown_id(mask)
        header_obj = self._schema.s_header_at_level(mask, shadow_level=shadow_level)
        header = json.dumps(header_obj) if not isinstance(header_obj, str) else header_obj

        # NL fallback (L4)
        if shadow_level >= 4:
            rows = []
            for resolved, text in zip(resolved_batch, texts):
                parts = [
                    f"{f}: {resolved.get(f, _ABSENT)}"
                    for f in active_fields
                    if resolved.get(f) is not None
                ]
                rows.append((", ".join(parts), text))
            return EncodedBatch(
                header=header, rows=rows, schema_id=schema_id,
                mask=mask, is_cutdown=is_cutdown,
            )

        # Positional arrays (L0-L3)
        rows = []
        for resolved, text in zip(resolved_batch, texts):
            row = [
                resolved.get(f) if resolved.get(f) is not None else _ABSENT
                for f in active_fields
            ]
            rows.append((json.dumps(row), text))

        return EncodedBatch(
            header=header, rows=rows, schema_id=schema_id,
            mask=mask, is_cutdown=is_cutdown,
        )

    def encode_metadata(
        self,
        chunk: dict[str, Any],
    ) -> dict[str, Any]:
        """Encode a single record's metadata as _dcp fields.

        Returns a dict with _dcp (positional array) and _dcp_schema keys,
        suitable for merging into the record's metadata dict.

        Used by framework adapters (LlamaIndex postprocessor, etc.).
        Absent fields are represented as '-'.
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
        row = [
            resolved.get(f) if resolved.get(f) is not None else _ABSENT
            for f in active_fields
        ]

        return {
            "_dcp": row,
            "_dcp_schema": self._schema.cutdown_id(mask),
        }