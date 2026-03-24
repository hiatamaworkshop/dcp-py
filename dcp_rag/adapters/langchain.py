"""LangChain adapter: DcpRunnable.

Usage:
    from dcp_rag.adapters.langchain import DcpRunnable

    chain = retriever | DcpRunnable.from_preset("qdrant") | prompt | llm
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from langchain_core.runnables import RunnableSerializable
    from langchain_core.documents import Document
except ImportError:
    raise ImportError(
        "langchain-core is required for LangChain adapter. "
        "Install with: pip install dcp-rag[langchain]"
    )

from dcp_rag.core.encoder import DcpEncoder


class DcpRunnable(RunnableSerializable[list[Document], list[Document]]):
    """LangChain Runnable that injects DCP metadata into Documents.

    Accepts list[Document] from a retriever and returns the same Documents
    with _dcp and _dcp_schema added to their metadata.
    """

    encoder: Any  # DcpEncoder, typed as Any for pydantic compat

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, encoder: DcpEncoder, **kwargs: Any):
        super().__init__(encoder=encoder, **kwargs)

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        *,
        schema: str = "rag-chunk-meta:v1",
        overrides: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> DcpRunnable:
        encoder = DcpEncoder.from_preset(
            preset_name, schema=schema, overrides=overrides
        )
        return cls(encoder=encoder, **kwargs)

    def invoke(
        self, input: list[Document], config: Any = None, **kwargs: Any
    ) -> list[Document]:
        for doc in input:
            source = {**doc.metadata}
            dcp_fields = self.encoder.encode_metadata(source)
            if dcp_fields:
                doc.metadata.update(dcp_fields)
        return input
