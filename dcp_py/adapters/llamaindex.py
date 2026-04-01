"""LlamaIndex adapter: DcpNodePostprocessor.

Usage:
    from dcp_py.adapters.llamaindex import DcpNodePostprocessor

    query_engine = index.as_query_engine(
        node_postprocessors=[DcpNodePostprocessor.from_preset("pinecone")]
    )
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
except ImportError:
    raise ImportError(
        "llama-index-core is required for LlamaIndex adapter. "
        "Install with: pip install dcp-rag[llamaindex]"
    )

from dcp_py.core.encoder import DcpEncoder


class DcpNodePostprocessor(BaseNodePostprocessor):
    """LlamaIndex node postprocessor that injects DCP metadata.

    Adds _dcp (positional array) and _dcp_schema to each node's metadata.
    Original metadata is preserved — DCP fields are additive.
    """

    _encoder: DcpEncoder

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, encoder: DcpEncoder, **kwargs: Any):
        super().__init__(**kwargs)
        self._encoder = encoder

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        *,
        schema: str = "rag-chunk-meta:v1",
        overrides: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> DcpNodePostprocessor:
        encoder = DcpEncoder.from_preset(
            preset_name, schema=schema, overrides=overrides
        )
        return cls(encoder=encoder, **kwargs)

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        for node_with_score in nodes:
            node = node_with_score.node
            # Build source dict from node metadata + score
            source = {**node.metadata, "score": node_with_score.score}
            dcp_fields = self._encoder.encode_metadata(source)
            if dcp_fields:
                node.metadata.update(dcp_fields)
        return nodes
