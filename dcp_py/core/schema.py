"""Layer 0: DCP Schema loader and validator.

Loads schema definitions from JSON files (schemas/*.json) and validates
DCP positional arrays against their schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


# Default schemas directory: dcp_py/schemas/ (inside the package)
_SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


@dataclass(frozen=True)
class FieldType:
    """Type definition for a single schema field."""

    type: str | list[str]
    description: str = ""
    enum: list[Any] | None = None
    min: float | None = None
    max: float | None = None

    def validate(self, value: Any) -> str | None:
        """Return error message if value is invalid, None if ok."""
        if value is None:
            # Check if null is allowed
            types = self.type if isinstance(self.type, list) else [self.type]
            if "null" in types:
                return None
            return "null not allowed"

        types = self.type if isinstance(self.type, list) else [self.type]
        type_ok = False
        for t in types:
            if t == "string" and isinstance(value, str):
                type_ok = True
            elif t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
                type_ok = True
            elif t == "boolean" and isinstance(value, bool):
                type_ok = True
            elif t == "null" and value is None:
                type_ok = True

        if not type_ok:
            return f"expected {self.type}, got {type(value).__name__}"

        if self.enum is not None and value not in self.enum:
            return f"value {value!r} not in enum {self.enum}"

        if self.min is not None and isinstance(value, (int, float)):
            if value < self.min:
                return f"value {value} < min {self.min}"

        if self.max is not None and isinstance(value, (int, float)):
            if value > self.max:
                return f"value {value} > max {self.max}"

        return None


@dataclass(frozen=True)
class DcpSchema:
    """A DCP schema definition.

    Attributes:
        id: Schema identifier (e.g. "rag-chunk-meta:v1")
        description: Human-readable description
        fields: Ordered list of field names
        field_count: Number of fields (must match len(fields))
        types: Field name → FieldType mapping
        examples: List of example positional arrays
        origin: Optional metadata about the data stream this schema targets.
                Dict with optional keys:
                  source    — free-form stream identifier (e.g. "tavily/search", "sensor/gyro")
                  direction — "input" | "output" | "bidirectional" (default: "bidirectional")
    """

    id: str
    description: str
    fields: tuple[str, ...]
    field_count: int
    types: dict[str, FieldType] = field(default_factory=dict)
    examples: tuple[tuple[Any, ...], ...] = ()
    origin: dict[str, str] | None = None
    nest_schemas: dict[str, Any] | None = None  # field → {schema: DcpSchemaDef, mapping: FieldMappingDef}

    @property
    def full_mask(self) -> int:
        """Bitmask with all field bits set."""
        return (1 << self.field_count) - 1

    def field_bit(self, field_name: str) -> int:
        """Return the bit value for a field (MSB = first field)."""
        idx = self.fields.index(field_name)
        return 1 << (self.field_count - 1 - idx)

    def fields_from_mask(self, mask: int) -> tuple[str, ...]:
        """Return field names corresponding to set bits in mask."""
        return tuple(
            f for i, f in enumerate(self.fields)
            if mask & (1 << (self.field_count - 1 - i))
        )

    def cutdown_id(self, mask: int) -> str:
        """Generate cutdown schema ID: base_id#hex_mask."""
        if mask == self.full_mask:
            return self.id
        return f"{self.id}#{mask:x}"

    def s_header(self, mask: int | None = None) -> list[Any]:
        """Generate $S header line as a list.

        Args:
            mask: Field presence bitmask. None = full schema.
        """
        if mask is None or mask == self.full_mask:
            return ["$S", self.id, self.field_count] + list(self.fields)
        active = self.fields_from_mask(mask)
        return ["$S", self.cutdown_id(mask), len(active)] + list(active)

    def s_header_at_level(
        self, mask: int | None = None, *, header_density: int = 2,
    ) -> str | list[Any] | dict[str, Any]:
        """Generate header at the specified shadow level.

        Shadow levels:
            0 — fields only: list of field names, no protocol info
            1 — with schema ID: field names + schema identifier
            2 — full protocol: $S + ID + field count + field names (default)
            3 — full schema definition: complete JSON schema object
            4 — NL fallback: natural language field descriptions

        Returns:
            Level 0-2: list (JSON-serializable)
            Level 3: dict (JSON-serializable schema definition)
            Level 4: str (natural language)
        """
        active = self.fields if (mask is None or mask == self.full_mask) else self.fields_from_mask(mask)
        sid = self.cutdown_id(mask) if mask is not None else self.id

        if header_density <= 0:
            # L0: fields only
            return list(active)

        if header_density == 1:
            # L1: with schema ID
            return ["$S", sid] + list(active)

        if header_density == 2:
            # L2: full protocol (original behavior)
            return self.s_header(mask)

        if header_density == 3:
            # L3: full schema definition
            types_dict: dict[str, Any] = {}
            for fname in active:
                ft = self.types.get(fname)
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
                "id": sid,
                "fields": list(active),
                "fieldCount": len(active),
                "types": types_dict,
            }

        # L4: NL fallback
        parts = []
        for fname in active:
            ft = self.types.get(fname)
            if ft and ft.description:
                parts.append(f"{fname}: {ft.description}")
            else:
                parts.append(fname)
        return "Fields: " + ", ".join(parts)

    def validate_row(self, row: list[Any], mask: int | None = None) -> list[str]:
        """Validate a positional array against this schema.

        Returns list of error messages (empty = valid).
        """
        errors = []
        active_fields = self.fields if mask is None else self.fields_from_mask(mask)

        if len(row) != len(active_fields):
            errors.append(
                f"expected {len(active_fields)} fields, got {len(row)}"
            )
            return errors

        for i, (fname, value) in enumerate(zip(active_fields, row)):
            ftype = self.types.get(fname)
            if ftype:
                err = ftype.validate(value)
                if err:
                    errors.append(f"field {fname} (pos {i}): {err}")

        return errors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DcpSchema:
        """Load schema from a parsed JSON dict."""
        if data.get("$dcp") != "schema":
            raise ValueError(f"not a DCP schema: missing or invalid $dcp marker")

        types = {}
        for fname, tdef in data.get("types", {}).items():
            types[fname] = FieldType(
                type=tdef["type"],
                description=tdef.get("description", ""),
                enum=tdef.get("enum"),
                min=tdef.get("min"),
                max=tdef.get("max"),
            )

        fields = tuple(data["fields"])
        examples = tuple(tuple(ex) for ex in data.get("examples", []))
        origin = data.get("origin")  # optional: {source, direction}
        nest_schemas = data.get("nestSchemas")  # optional: {field: {schema, mapping}}

        return cls(
            id=data["id"],
            description=data.get("description", ""),
            fields=fields,
            field_count=data["fieldCount"],
            types=types,
            examples=examples,
            origin=origin,
            nest_schemas=nest_schemas,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> DcpSchema:
        """Load schema from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class SchemaRegistry:
    """Registry of loaded DCP schemas.

    Schemas are loaded from a directory of JSON files.
    """

    def __init__(self, schemas_dir: str | Path | None = None):
        self._schemas: dict[str, DcpSchema] = {}
        if schemas_dir is not None:
            self.load_dir(schemas_dir)

    def load_dir(self, schemas_dir: str | Path) -> None:
        """Load all .json schema files from a directory."""
        d = Path(schemas_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"schemas directory not found: {d}")
        for p in sorted(d.glob("*.json")):
            try:
                schema = DcpSchema.from_file(p)
                self._schemas[schema.id] = schema
            except (ValueError, KeyError):
                continue  # skip non-schema JSON files

    def get(self, schema_id: str) -> DcpSchema:
        """Get a schema by ID. Raises KeyError if not found."""
        try:
            return self._schemas[schema_id]
        except KeyError:
            available = ", ".join(sorted(self._schemas.keys()))
            raise KeyError(
                f"schema {schema_id!r} not found. "
                f"Available: [{available}]"
            ) from None

    def __contains__(self, schema_id: str) -> bool:
        return schema_id in self._schemas

    def list(self) -> list[str]:
        """Return sorted list of loaded schema IDs."""
        return sorted(self._schemas.keys())


def load_default_registry() -> SchemaRegistry:
    """Load the default schema registry from schemas/ directory."""
    return SchemaRegistry(_SCHEMAS_DIR)
