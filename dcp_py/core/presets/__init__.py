"""Preset FieldMappings for common data sources.

Structure:
  presets/rag/    — Vector DB presets (Pinecone, Qdrant, Weaviate, Chroma, Milvus)
  presets/sql/    — SQL / DataFrame presets (psycopg2, sqlalchemy, sqlite3)
  presets/log/    — Structured log presets (cloudwatch, datadog, loki)

Top-level get_preset() and list_presets() delegate to rag/ by default
for backward compatibility.
"""

from dcp_py.core.presets.rag.registry import get_preset, list_presets
from dcp_py.core.presets.log.registry import get_log_preset, list_log_presets
from dcp_py.core.presets.sql.registry import get_sql_preset, list_sql_presets

__all__ = ["get_preset", "list_presets", "get_log_preset", "list_log_presets", "get_sql_preset", "list_sql_presets"]