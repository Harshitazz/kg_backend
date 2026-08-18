"""Compatibility layer for the knowledge-graph service.

The implementation lives under services/kg/*. This module keeps the original import
surface stable for controllers and other modules while the logic is split into
smaller responsibilities.
"""

from kg_backend.utils.kg.extraction import extract_knowledge_graph, set_llm
from kg_backend.utils.kg.query import (
    extract_llm_text,
    get_knowledge_graph_history,
    get_node_explanation,
    get_relevant_nodes_for_question,
    get_user_knowledge_graph,
    get_user_task_ids,
    query_kg_for_question,
)
from kg_backend.utils.kg.storage import (
    get_neo4j_driver,
    process_documents_for_kg,
    process_text_for_kg,
    store_knowledge_graph,
)

__all__ = [
    "set_llm",
    "extract_knowledge_graph",
    "get_neo4j_driver",
    "store_knowledge_graph",
    "process_documents_for_kg",
    "process_text_for_kg",
    "get_user_knowledge_graph",
    "get_user_task_ids",
    "get_knowledge_graph_history",
    "extract_llm_text",
    "get_node_explanation",
    "query_kg_for_question",
    "get_relevant_nodes_for_question",
]
