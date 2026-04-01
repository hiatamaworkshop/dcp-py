"""Haystack adapter: DcpComponent.

Usage:
    from dcp_py.adapters.haystack import DcpComponent

    pipeline.add_component("dcp", DcpComponent.from_preset("weaviate"))
    pipeline.connect("retriever.documents", "dcp.documents")
    pipeline.connect("dcp.documents", "prompt_builder.documents")
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from haystack import Document, component
except ImportError:
    raise ImportError(
        "haystack-ai is required for Haystack adapter. "
        "Install with: pip install dcp-rag[haystack]"
    )

from dcp_py.core.encoder import DcpEncoder


@component
class DcpComponent:
    """Haystack component that injects DCP metadata into Documents.

    Input: list[Document] from retriever
    Output: list[Document] with _dcp and _dcp_schema in metadata
    """

    def __init__(self, encoder: DcpEncoder):
        self._encoder = encoder

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        *,
        schema: str = "rag-chunk-meta:v1",
        overrides: Optional[dict[str, str]] = None,
    ) -> DcpComponent:
        encoder = DcpEncoder.from_preset(
            preset_name, schema=schema, overrides=overrides
        )
        return cls(encoder=encoder)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        for doc in documents:
            source = {**(doc.meta or {})}
            if doc.score is not None:
                source["score"] = doc.score
            dcp_fields = self._encoder.encode_metadata(source)
            if dcp_fields:
                if doc.meta is None:
                    doc.meta = {}
                doc.meta.update(dcp_fields)
        return {"documents": documents}
