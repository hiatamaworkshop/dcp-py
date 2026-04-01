"""Preset FieldMappings for common data sources.

Structure:
  presets/rag/    — Vector DB presets (Pinecone, Qdrant, Weaviate, Chroma, Milvus)
  presets/sql/    — SQL / DataFrame presets (planned)
  presets/log/    — Structured log presets (planned)

Top-level get_preset() and list_presets() delegate to rag/ by default
for backward compatibility.
"""

from dcp_py.core.presets.rag.registry import get_preset, list_presets

__all__ = ["get_preset", "list_presets"]