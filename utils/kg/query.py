import json
import logging
import re
from typing import Any, Dict, List, Optional

from kg_backend.utils.kg.storage import get_neo4j_driver
from services.qdrant_service import get_task_collections, query_vectorstore_multi_task

logger = logging.getLogger(__name__)

llm = None


def set_llm(llm_instance):
    global llm
    llm = llm_instance


def extract_llm_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        text = content.get("content")
        if isinstance(text, str):
            return text
        text = content.get("explanation")
        if isinstance(text, str):
            return text
    return str(content)


def get_user_knowledge_graph(user_id: str, task_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    try:
        driver = get_neo4j_driver()
    except Exception as exc:
        logger.error(f"Cannot connect to Neo4j: {str(exc)}")
        return {"nodes": [], "links": []}

    try:
        with driver.session() as session:
            if task_id:
                query = """
                MATCH (n)
                WHERE n.user_id = $user_id AND n.task_id = $task_id AND n.name IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                WHERE degree >= 1
                RETURN DISTINCT n.name as name, labels(n) as labels, COALESCE(n.type, head(labels(n))) as type, properties(n) as properties, degree
                ORDER BY degree DESC
                LIMIT $limit
                """
                result = session.run(query, user_id=user_id, task_id=task_id, limit=limit)
            else:
                query = """
                MATCH (n)
                WHERE n.user_id = $user_id AND n.name IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                WHERE degree >= 1
                RETURN DISTINCT n.name as name, labels(n) as labels, COALESCE(n.type, head(labels(n))) as type, properties(n) as properties, degree
                ORDER BY degree DESC
                LIMIT $limit
                """
                result = session.run(query, user_id=user_id, limit=limit)

            nodes = []
            node_map = {}
            for record in result:
                name = record["name"]
                labels_list = record["labels"]
                node_type = record.get("type") or (labels_list[0] if labels_list else "OTHER")
                properties = dict(record["properties"]) if record["properties"] else {}
                degree = record.get("degree", 0)
                node_id = len(nodes)
                node_map[name] = node_id
                nodes.append({"id": node_id, "name": name, "type": node_type, "label": name, "properties": properties, "degree": degree})

            links = []
            if nodes:
                node_names = [node["name"] for node in nodes]
                if task_id:
                    rel_query = """
                    MATCH (source)-[r]->(target)
                    WHERE source.user_id = $user_id AND target.user_id = $user_id AND source.task_id = $task_id AND target.task_id = $task_id AND source.name IN $node_names AND target.name IN $node_names
                    RETURN source.name as source, target.name as target, type(r) as relationship_type, properties(r) as properties
                    """
                    rel_result = session.run(rel_query, user_id=user_id, task_id=task_id, node_names=node_names)
                else:
                    rel_query = """
                    MATCH (source)-[r]->(target)
                    WHERE source.user_id = $user_id AND target.user_id = $user_id AND source.name IN $node_names AND target.name IN $node_names
                    RETURN source.name as source, target.name as target, type(r) as relationship_type, properties(r) as properties
                    """
                    rel_result = session.run(rel_query, user_id=user_id, node_names=node_names)

                for record in rel_result:
                    source_name = record["source"]
                    target_name = record["target"]
                    rel_type = record["relationship_type"]
                    properties = dict(record["properties"]) if record["properties"] else {}
                    if source_name in node_map and target_name in node_map:
                        links.append({"source": node_map[source_name], "target": node_map[target_name], "type": rel_type, "label": rel_type, "properties": properties})

            connected_node_ids = set()
            for link in links:
                connected_node_ids.add(link["source"])
                connected_node_ids.add(link["target"])

            if len(nodes) <= 20:
                final_nodes = nodes
            else:
                final_nodes = [node for node in nodes if node["id"] in connected_node_ids]

            final_node_map = {}
            for idx, node in enumerate(final_nodes):
                final_node_map[node["name"]] = idx
                node["id"] = idx

            final_links = []
            for link in links:
                source_name = nodes[link["source"]]["name"] if link["source"] < len(nodes) else None
                target_name = nodes[link["target"]]["name"] if link["target"] < len(nodes) else None
                if source_name in final_node_map and target_name in final_node_map:
                    final_links.append({"source": final_node_map[source_name], "target": final_node_map[target_name], "type": link["type"], "label": link["label"], "properties": link["properties"]})

            logger.info(f"Returning {len(final_nodes)} nodes and {len(final_links)} links (filtered from {len(nodes)} nodes)")
            return {"nodes": final_nodes, "links": final_links}
    except Exception as exc:
        logger.error(f"Error retrieving knowledge graph: {str(exc)}")
        return {"nodes": [], "links": []}
    finally:
        driver.close()


def get_user_task_ids(user_id: str) -> List[str]:
    try:
        driver = get_neo4j_driver()
    except Exception as exc:
        logger.error(f"Cannot connect to Neo4j: {str(exc)}")
        return []

    try:
        with driver.session() as session:
            query = """
            MATCH (n)
            WHERE n.user_id = $user_id AND n.task_id IS NOT NULL
            RETURN DISTINCT n.task_id as task_id
            ORDER BY task_id DESC
            """
            result = session.run(query, user_id=user_id)
            return [record["task_id"] for record in result]
    except Exception as exc:
        logger.error(f"Error retrieving task IDs: {str(exc)}")
        return []
    finally:
        driver.close()


def get_knowledge_graph_history(user_id: str) -> List[Dict[str, Any]]:
    try:
        driver = get_neo4j_driver()
    except Exception as exc:
        logger.error(f"Cannot connect to Neo4j: {str(exc)}")
        return []

    try:
        with driver.session() as session:
            query = """
            MATCH (n)
            WHERE n.user_id = $user_id AND n.task_id IS NOT NULL
            WITH n.task_id as task_id, collect(DISTINCT n.source) as sources, count(DISTINCT n) as node_count
            MATCH (a)-[r]->(b)
            WHERE a.user_id = $user_id AND a.task_id = task_id
            WITH task_id, sources, node_count, count(DISTINCT r) as rel_count
            RETURN task_id, sources[0] as source, node_count, rel_count, task_id as created_at
            ORDER BY task_id DESC
            """
            result = session.run(query, user_id=user_id)
            history = []
            for record in result:
                history.append({
                    "task_id": record["task_id"],
                    "source": record["source"] or "Unknown",
                    "node_count": record["node_count"],
                    "relationship_count": record["rel_count"],
                    "created_at": record["created_at"],
                })
            return history
    except Exception as exc:
        logger.error(f"Error retrieving KG history: {str(exc)}")
        return []
    finally:
        driver.close()


def get_node_explanation(node_name: str, user_id: str, task_id: Optional[str] = None) -> str:
    if llm is None:
        return "Explanation not available (LLM not initialized)"

    embedding_context = ""
    try:
        task_ids_to_query = []
        if task_id:
            task_ids_to_query = [task_id]
        else:
            collections = get_task_collections(user_id)
            task_ids_to_query = [coll["task_id"] for coll in collections]

        if task_ids_to_query:
            relevant_docs = query_vectorstore_multi_task(question=node_name, user_id=user_id, task_ids=task_ids_to_query, k=5)
            if relevant_docs:
                context_texts = []
                for doc in relevant_docs:
                    text = doc.page_content.strip()
                    if node_name.lower() in text.lower():
                        context_texts.append(text[:500])
                if context_texts:
                    embedding_context = "\n\nRelevant context from documents:\n"
                    for i, text in enumerate(context_texts[:3], 1):
                        embedding_context += f"{i}. {text}\n"
    except Exception as exc:
        logger.warning(f"Could not retrieve embedding context: {str(exc)}")

    try:
        driver = get_neo4j_driver()
    except Exception as exc:
        logger.error(f"Cannot connect to Neo4j: {str(exc)}")
        return "Explanation not available"

    try:
        with driver.session() as session:
            if task_id:
                query = """
                MATCH (n {name: $node_name, user_id: $user_id, task_id: $task_id})
                OPTIONAL MATCH (n)-[r1]->(connected)
                OPTIONAL MATCH (connected)-[r2]->(n)
                RETURN n, collect(DISTINCT {rel: type(r1), target: connected.name, target_type: connected.type}) as outgoing, collect(DISTINCT {rel: type(r2), source: connected.name, source_type: connected.type}) as incoming
                LIMIT 1
                """
                result = session.run(query, node_name=node_name, user_id=user_id, task_id=task_id)
            else:
                query = """
                MATCH (n {name: $node_name, user_id: $user_id})
                OPTIONAL MATCH (n)-[r1]->(connected)
                OPTIONAL MATCH (connected)-[r2]->(n)
                RETURN n, collect(DISTINCT {rel: type(r1), target: connected.name, target_type: connected.type}) as outgoing, collect(DISTINCT {rel: type(r2), source: connected.name, source_type: connected.type}) as incoming
                LIMIT 1
                """
                result = session.run(query, node_name=node_name, user_id=user_id)

            record = result.single()
            if not record:
                return f"No information found about '{node_name}'"

            node = record["n"]
            node_type = node.get("type", "entity")
            source = node.get("source", "unknown")
            outgoing = [r for r in record["outgoing"] if r.get("target")]
            incoming = [r for r in record["incoming"] if r.get("source")]

            context = f"Entity: {node_name}\nType: {node_type}\nSource: {source}\n"
            if outgoing:
                context += "\nOutgoing connections (what this entity relates to):\n"
                for rel in outgoing[:10]:
                    target_name = rel.get("target", "")
                    target_type = rel.get("target_type", "")
                    rel_type = rel.get("rel", "")
                    if target_name:
                        type_info = f" ({target_type})" if target_type else ""
                        context += f"- {rel_type} -> {target_name}{type_info}\n"
            if incoming:
                context += "\nIncoming connections (what relates to this entity):\n"
                for rel in incoming[:10]:
                    source_name = rel.get("source", "")
                    source_type = rel.get("source_type", "")
                    rel_type = rel.get("rel", "")
                    if source_name:
                        type_info = f" ({source_type})" if source_type else ""
                        context += f"- {source_name}{type_info} {rel_type} -> {node_name}\n"

            full_context = context + embedding_context
            prompt = f"""You are explaining an entity from a knowledge graph. Provide a comprehensive, context-aware explanation that:
1. Describes what the entity is based on the document context provided
2. Explains its role and significance in the knowledge graph based on its connections
3. Synthesizes information from both the document content and graph relationships
4. Be informative and educational (3-5 sentences)

Entity Information:
{full_context}

Provide a clear, context-rich explanation that combines the document context with the graph structure:"""
            response = llm.invoke(prompt).content
            return extract_llm_text(response)
    except Exception as exc:
        logger.error(f"Error generating node explanation: {str(exc)}")
        return f"Error generating explanation: {str(exc)}"
    finally:
        driver.close()


def query_kg_for_question(question: str, user_id: str, task_ids: Optional[List[str]] = None, max_entities: int = 10) -> str:
    if llm is None:
        logger.debug("LLM not initialized, skipping KG query")
        return ""

    try:
        driver = get_neo4j_driver()
    except Exception as exc:
        logger.error(f"Cannot connect to Neo4j: {str(exc)}")
        return ""

    try:
        with driver.session() as session:
            entity_extraction_prompt = f"""Extract key entities, concepts, or topics from this question that might be in a knowledge graph.

Question: {question}

Return a JSON list of 3-5 key terms (entities, concepts, topics) that should be searched in the knowledge graph.
Format: ["term1", "term2", "term3"]

Terms:"""
            try:
                response = llm.invoke(entity_extraction_prompt)
                entity_result = extract_llm_text(response.content)
                json_match = re.search(r'\[.*?\]', entity_result, re.DOTALL)
                if json_match:
                    search_terms = json.loads(json_match.group())
                else:
                    search_terms = re.findall(r'"([^"]+)"', entity_result)
                    if not search_terms:
                        search_terms = [w for w in question.split() if len(w) > 3][:5]
            except Exception as exc:
                logger.warning(f"Error extracting entities from question: {str(exc)}")
                search_terms = [w for w in question.split() if len(w) > 3][:5]

            if not search_terms:
                logger.debug("No search terms extracted from question")
                return ""

            if task_ids:
                task_filter = "AND n.task_id IN $task_ids"
                connected_task_filter = "AND connected.task_id IN $task_ids"
                params = {"user_id": user_id, "task_ids": task_ids, "search_terms": search_terms}
            else:
                task_filter = ""
                connected_task_filter = ""
                params = {"user_id": user_id, "search_terms": search_terms}

            query = f"""
            MATCH (n:Entity)
            WHERE n.user_id = $user_id
              {task_filter}
              AND (
                ANY(term IN $search_terms WHERE toLower(n.name) CONTAINS toLower(term))
                OR ANY(term IN $search_terms WHERE toLower(term) CONTAINS toLower(n.name))
              )
            WITH n,
                 [term IN $search_terms WHERE toLower(n.name) CONTAINS toLower(term) OR toLower(term) CONTAINS toLower(n.name) | term] as matches
            ORDER BY size(matches) DESC
            LIMIT $max_entities
            OPTIONAL MATCH (n)-[r1]->(connected)
            WHERE connected.user_id = $user_id {connected_task_filter}
            OPTIONAL MATCH (connected)-[r2]->(n)
            WHERE connected.user_id = $user_id {connected_task_filter}
            RETURN DISTINCT n.name as name,
                   COALESCE(n.type, 'OTHER') as type,
                   collect(DISTINCT {{rel: type(r1), target: connected.name}})[0..3] as outgoing,
                   collect(DISTINCT {{rel: type(r2), source: connected.name}})[0..3] as incoming
            """
            params["max_entities"] = max_entities
            result = session.run(query, params)

            entities_context = []
            for record in result:
                name = record["name"]
                entity_type = record.get("type", "entity")
                outgoing = [r for r in record["outgoing"] if r and r.get("target")]
                incoming = [r for r in record["incoming"] if r and r.get("source")]

                context_parts = [f"Entity: {name} (Type: {entity_type})"]
                if outgoing:
                    context_parts.append("Connections:")
                    for rel in outgoing[:3]:
                        if rel.get("rel") and rel.get("target"):
                            context_parts.append(f"  - {name} {rel['rel']} {rel['target']}")
                if incoming:
                    for rel in incoming[:3]:
                        if rel.get("rel") and rel.get("source"):
                            context_parts.append(f"  - {rel['source']} {rel['rel']} {name}")
                entities_context.append("\n".join(context_parts))

            if entities_context:
                return "\n\nKnowledge Graph Context:\n" + "\n\n".join(entities_context[:5])
            return ""
    except Exception as exc:
        logger.error(f"Error querying KG for question: {str(exc)}")
        import traceback
        logger.error(traceback.format_exc())
        return ""
    finally:
        driver.close()


def get_relevant_nodes_for_question(question: str, user_id: str, task_ids: Optional[List[str]] = None, max_nodes: int = 10) -> List[Dict[str, Any]]:
    if llm is None:
        return []

    try:
        driver = get_neo4j_driver()
    except Exception as exc:
        logger.error(f"Cannot connect to Neo4j: {str(exc)}")
        return []

    try:
        with driver.session() as session:
            entity_extraction_prompt = f"""Extract key entities, concepts, or topics from this question that might be in a knowledge graph.

Question: {question}

Return a JSON list of 3-5 key terms (entities, concepts, topics) that should be searched in the knowledge graph.
Format: ["term1", "term2", "term3"]

Terms:"""
            try:
                response = llm.invoke(entity_extraction_prompt)
                entity_result = extract_llm_text(response.content)
                json_match = re.search(r'\[.*?\]', entity_result, re.DOTALL)
                if json_match:
                    search_terms = json.loads(json_match.group())
                else:
                    search_terms = re.findall(r'"([^"]+)"', entity_result)
                    if not search_terms:
                        search_terms = [w for w in question.split() if len(w) > 3][:5]
            except Exception as exc:
                logger.warning(f"Error extracting entities from question: {str(exc)}")
                search_terms = [w for w in question.split() if len(w) > 3][:5]

            if not search_terms:
                return []

            if task_ids:
                task_filter = "AND n.task_id IN $task_ids"
                params = {"user_id": user_id, "task_ids": task_ids, "search_terms": search_terms}
            else:
                task_filter = ""
                params = {"user_id": user_id, "search_terms": search_terms}

            query = f"""
            MATCH (n:Entity)
            WHERE n.user_id = $user_id
              {task_filter}
              AND (
                ANY(term IN $search_terms WHERE toLower(n.name) CONTAINS toLower(term))
                OR ANY(term IN $search_terms WHERE toLower(term) CONTAINS toLower(n.name))
              )
            WITH n,
                 [term IN $search_terms WHERE toLower(n.name) CONTAINS toLower(term) OR toLower(term) CONTAINS toLower(n.name) | term] as matches
            ORDER BY size(matches) DESC
            LIMIT $max_nodes
            RETURN DISTINCT n.name as name, COALESCE(n.type, 'OTHER') as type, properties(n) as properties
            """
            params["max_nodes"] = max_nodes
            result = session.run(query, params)

            nodes = []
            for record in result:
                nodes.append({
                    "name": record["name"],
                    "type": record.get("type", "OTHER"),
                    "properties": dict(record["properties"]) if record["properties"] else {},
                })
            return nodes
    except Exception as exc:
        logger.error(f"Error getting relevant nodes: {str(exc)}")
        return []
    finally:
        driver.close()
