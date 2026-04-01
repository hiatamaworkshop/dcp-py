"""Output Controller — place LLM key-value output into DCP positional arrays.

The controller separates meaning determination (LLM) from structural placement
(system). The LLM outputs key-value pairs, the controller arranges them into
schema-compliant positional arrays. No semantic inference — just placement.

This is the DCP equivalent of a form/GUI for humans: the LLM fills in fields,
the system enforces structure.

Usage:
    ctrl = OutputController(schema="knowledge:v1")
    row = ctrl.place({"action": "replace", "domain": "auth", "detail": "jwt migration", "confidence": 0.9})
    # → ["replace", "auth", "jwt migration", 0.9]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dcp_py.core.schema import DcpSchema, SchemaRegistry, load_default_registry


@dataclass
class PlacementResult:
    """Result of placing key-value data into a DCP positional array."""

    row: list[Any]
    schema_id: str
    warnings: list[str]
    is_valid: bool

    def to_json(self) -> str:
        """Serialize the positional array to JSON."""
        return json.dumps(self.row)


class OutputController:
    """Place LLM key-value output into schema-ordered positional arrays.

    The LLM decides meaning (which field has which value).
    The controller decides placement (which position each value goes in).
    No semantic inference — if the LLM says action="replace", the controller
    puts "replace" at the position defined by the schema. That's it.
    """

    def __init__(
        self,
        schema: str | DcpSchema,
        *,
        registry: SchemaRegistry | None = None,
        strict: bool = False,
    ):
        """Initialize controller.

        Args:
            schema: Schema ID string or DcpSchema instance
            registry: Schema registry. Uses default if None.
            strict: If True, raise ValueError on validation errors.
                    If False, return warnings in PlacementResult.
        """
        if registry is None:
            registry = load_default_registry()

        if isinstance(schema, str):
            self._schema = registry.get(schema)
        else:
            self._schema = schema

        self._strict = strict

    @property
    def schema(self) -> DcpSchema:
        return self._schema

    @property
    def fields(self) -> tuple[str, ...]:
        return self._schema.fields

    def place(self, data: dict[str, Any]) -> PlacementResult:
        """Place key-value data into a positional array.

        Args:
            data: Dict of field_name → value from LLM output.
                  Extra keys are silently ignored.
                  Missing keys become None in the output.

        Returns:
            PlacementResult with positional array, validation warnings, etc.
        """
        warnings: list[str] = []

        # Check for unknown keys
        known_fields = set(self._schema.fields)
        extra_keys = set(data.keys()) - known_fields
        if extra_keys:
            warnings.append(f"ignored unknown keys: {sorted(extra_keys)}")

        # Check for missing keys
        missing_keys = known_fields - set(data.keys())
        if missing_keys:
            warnings.append(f"missing fields (set to null): {sorted(missing_keys)}")

        # Build positional array in schema order
        row: list[Any] = []
        for field_name in self._schema.fields:
            row.append(data.get(field_name))

        # Validate against schema types
        validation_errors = self._schema.validate_row(row)
        if validation_errors:
            for err in validation_errors:
                warnings.append(f"validation: {err}")

        is_valid = len(validation_errors) == 0

        if self._strict and not is_valid:
            raise ValueError(
                f"DCP validation failed: {'; '.join(validation_errors)}"
            )

        return PlacementResult(
            row=row,
            schema_id=self._schema.id,
            warnings=warnings,
            is_valid=is_valid,
        )

    def place_batch(self, items: list[dict[str, Any]]) -> list[PlacementResult]:
        """Place multiple items. Convenience wrapper around place()."""
        return [self.place(item) for item in items]