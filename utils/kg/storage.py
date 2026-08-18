import logging
from typing import Any, Dict, List, Optional

from difflib import SequenceMatcher

from langchain_core.documents import Document
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

NEO4J_URI = __import__("os").getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = __import__("os").getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = __import__("os").getenv("NEO4J_PASSWORD", "password")


def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as exc:
        logger.error(f"Failed to connect to Neo4j: {str(exc)}")
        raise


def store_knowledge_graph(
    entities: List[Dict],
    relationships: List[Dict],
    user_id: str,
    source: str,
    task_id: Optional[str] = None,
    driver: Optional[Any] = None,
) -> None:
    if not entities and not relationships:
        logger.info("No entities or relationships to store")
        return

    should_close = False
    if driver is None:
        try:
            driver = get_neo4j_driver()
            should_close = True
        except Exception as exc:
            logger.error(f"Cannot connect to Neo4j: {str(exc)}")
            return

    try:
        with driver.session() as session:
            entity_count = 0
            for entity in entities:
                entity_name = entity.get("name", "").strip()
                if not entity_name:
                    continue

                entity_type = entity.get("type", "OTHER")
                properties = entity.get("properties", {}).copy()
                properties["user_id"] = user_id
                properties["source"] = source
                properties["type"] = entity_type
                if task_id:
                    properties["task_id"] = task_id

                check_query = """
                MATCH (e:Entity {name: $name, user_id: $user_id})
                WHERE ($task_id IS NULL OR e.task_id = $task_id)
                RETURN e.name as name, e.task_id as existing_task_id
                LIMIT 1
                """
                existing = session.run(check_query, name=entity_name, user_id=user_id, task_id=task_id).single()

                if existing:
                    update_query = f"""
                    MATCH (e:Entity {{name: $name, user_id: $user_id}})
                    WHERE ($task_id IS NULL OR e.task_id = $task_id)
                    SET e:`{entity_type}`
                    SET e += $properties
                    RETURN e
                    """
                    try:
                        result = session.run(update_query, name=entity_name, user_id=user_id, task_id=task_id, properties=properties)
                        result.consume()
                        entity_count += 1
                        continue
                    except Exception as exc:
                        logger.warning(f"Failed to update entity {entity_name}: {str(exc)}")

                similarity_query = """
                MATCH (e:Entity {user_id: $user_id})
                WHERE ($task_id IS NULL OR e.task_id = $task_id)
                RETURN e.name as name
                """
                existing_names = [record["name"] for record in session.run(similarity_query, user_id=user_id, task_id=task_id)]

                similar_entity = None
                for existing_name in existing_names:
                    similarity = SequenceMatcher(None, entity_name.lower(), existing_name.lower()).ratio()
                    if similarity >= 0.90:
                        similar_entity = existing_name
                        logger.info(f"Found similar entity: '{entity_name}' similar to '{existing_name}' (similarity: {similarity:.2f})")
                        break
                    if len(entity_name) > 5 and len(existing_name) > 5:
                        if entity_name.lower() in existing_name.lower() or existing_name.lower() in entity_name.lower():
                            shorter = min(len(entity_name), len(existing_name))
                            longer = max(len(entity_name), len(existing_name))
                            if shorter / longer >= 0.80:
                                similar_entity = existing_name
                                logger.info(f"Found substring match: '{entity_name}' matches '{existing_name}'")
                                break

                if similar_entity:
                    merge_query = f"""
                    MATCH (e:Entity {{name: $similar_name, user_id: $user_id}})
                    WHERE ($task_id IS NULL OR e.task_id = $task_id)
                    SET e:`{entity_type}`
                    SET e += $properties
                    RETURN e
                    """
                    try:
                        result = session.run(merge_query, similar_name=similar_entity, user_id=user_id, task_id=task_id, properties=properties)
                        result.consume()
                        entity_count += 1
                        continue
                    except Exception as exc:
                        logger.warning(f"Failed to merge similar entity {entity_name} with {similar_entity}: {str(exc)}")

                query = f"""
                MERGE (e:Entity {{name: $name, user_id: $user_id, task_id: $task_id}})
                SET e:`{entity_type}`
                SET e += $properties
                RETURN e
                """
                try:
                    result = session.run(query, name=entity_name, user_id=user_id, task_id=task_id, properties=properties)
                    result.consume()
                    entity_count += 1
                except Exception as exc:
                    logger.warning(f"Failed to set label {entity_type} for entity {entity_name}, using fallback: {str(exc)}")
                    try:
                        fallback_query = """
                        MERGE (e:Entity {name: $name, user_id: $user_id, task_id: $task_id})
                        SET e += $properties
                        RETURN e
                        """
                        result = session.run(fallback_query, name=entity_name, user_id=user_id, task_id=task_id, properties=properties)
                        result.consume()
                        entity_count += 1
                    except Exception as exc2:
                        logger.error(f"Failed to create entity {entity_name}: {str(exc2)}")
                        continue

            rel_count = 0
            for rel in relationships:
                source_entity = rel.get("source", "").strip()
                target_entity = rel.get("target", "").strip()
                rel_type = rel.get("type", "RELATED_TO")
                properties = rel.get("properties", {}).copy()
                properties["user_id"] = user_id
                properties["source"] = source
                if task_id:
                    properties["task_id"] = task_id

                if not source_entity or not target_entity:
                    continue

                query = f"""
                MATCH (source:Entity {{name: $source_name, user_id: $user_id, task_id: $task_id}})
                MATCH (target:Entity {{name: $target_name, user_id: $user_id, task_id: $task_id}})
                MERGE (source)-[r:{rel_type}]->(target)
                SET r += $properties
                RETURN r
                """

                try:
                    result = session.run(
                        query,
                        source_name=source_entity,
                        target_name=target_entity,
                        user_id=user_id,
                        task_id=task_id,
                        properties=properties,
                    )
                    result.consume()
                    rel_count += 1
                except Exception as exc:
                    logger.warning(f"Failed to create relationship {source_entity} -> {target_entity}: {str(exc)}")
                    continue

            logger.info(f"Stored {entity_count}/{len(entities)} entities and {rel_count}/{len(relationships)} relationships for source: {source}")
    except Exception as exc:
        logger.error(f"Error storing knowledge graph: {str(exc)}")
    finally:
        if should_close:
            driver.close()


def process_documents_for_kg(
    documents: List[Document],
    user_id: str,
    source: str,
    task_id: Optional[str] = None,
    chunk_size: int = 2000,
) -> None:
    if not documents:
        return

    try:
        driver = get_neo4j_driver()
    except Exception as exc:
        logger.warning(f"Neo4j not available, skipping knowledge graph creation: {str(exc)}")
        return

    all_entities = []
    all_relationships = []

    try:
        chunk_count = 0
        for doc in documents:
            text = doc.page_content
            if not text.strip():
                continue

            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count}/{len(documents)} for KG extraction from {source}")

            from kg_backend.utils.kg.extraction import extract_knowledge_graph

            kg_data = extract_knowledge_graph(text, source=source)
            entities = kg_data.get("entities", [])
            relationships = kg_data.get("relationships", [])

            logger.info(f"Chunk {chunk_count}: Extracted {len(entities)} entities, {len(relationships)} relationships")
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        logger.info(f"Total extracted: {len(all_entities)} entities, {len(all_relationships)} relationships from {source}")
        if all_entities or all_relationships:
            logger.info(f"Storing knowledge graph for {source} (task_id: {task_id})...")
            store_knowledge_graph(all_entities, all_relationships, user_id, source, task_id=task_id, driver=driver)
            logger.info(f"Knowledge graph storage completed for {source} (task_id: {task_id})")
        else:
            logger.warning(f"No entities or relationships extracted from {source}")
    except Exception as exc:
        logger.error(f"Error processing documents for knowledge graph: {str(exc)}")
    finally:
        driver.close()


def process_text_for_kg(text: str, user_id: str, source: str, task_id: Optional[str] = None) -> None:
    if not text or not text.strip():
        return

    try:
        driver = get_neo4j_driver()
    except Exception as exc:
        logger.warning(f"Neo4j not available, skipping knowledge graph creation: {str(exc)}")
        return

    try:
        from kg_backend.utils.kg.extraction import extract_knowledge_graph

        kg_data = extract_knowledge_graph(text, source=source)
        entities = kg_data.get("entities", [])
        relationships = kg_data.get("relationships", [])

        if entities or relationships:
            store_knowledge_graph(entities, relationships, user_id, source, task_id=task_id, driver=driver)
    except Exception as exc:
        logger.error(f"Error processing text for knowledge graph: {str(exc)}")
    finally:
        driver.close()
