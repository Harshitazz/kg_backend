"""
Knowledge Graph Service for extracting and storing entities and relationships in Neo4j
"""
import os
import json
import logging
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Global LLM instance
llm = None

def set_llm(llm_instance):
    """Set the LLM instance from main.py"""
    global llm
    llm = llm_instance

def get_neo4j_driver():
    """Get Neo4j driver instance"""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Verify connection
        driver.verify_connectivity()
        return driver
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        raise

# Knowledge graph extraction prompt - dynamic based on content type
KG_EXTRACTION_PROMPT = """Analyze the following text and determine its type (educational/academic, financial/budget, technical, business, or general).
Then extract entities and relationships appropriate for that content type.

For EDUCATIONAL/ACADEMIC content:
- Extract: Topics, Concepts, Theories, Methods, Applications, Researchers, Institutions
- Relationships: PART_OF, PREREQUISITE_FOR, USES, IMPLEMENTS, DERIVES_FROM, TRANSFORMS_INTO, RELATED_TO

For FINANCIAL/BUDGET content:
- Extract: Budget Items, Categories, Departments, Projects, Costs, Allocations, Time Periods
- Relationships: CONTAINS, ALLOCATED_TO, EXCEEDS, BELONGS_TO, FUNDS, OCCURS_IN

For TECHNICAL content:
- Extract: Technologies, Tools, Frameworks, Systems, Components, Standards
- Relationships: DEPENDS_ON, IMPLEMENTS, USES, COMPATIBLE_WITH, VERSION_OF

For BUSINESS content:
- Extract: Departments, Roles, Processes, Products, Services, Metrics
- Relationships: MANAGES, REPORTS_TO, PRODUCES, MEASURES, IMPACTS

For GENERAL content:
- Extract: Key Concepts, People, Organizations, Locations, Events
- Relationships: RELATED_TO, INVOLVES, LOCATED_IN, OCCURS_AT

Return a JSON object with the following structure:
{{
    "content_type": "educational|financial|technical|business|general",
    "entities": [
        {{
            "name": "entity name",
            "type": "TOPIC|CONCEPT|THEORY|METHOD|PERSON|ORGANIZATION|BUDGET_ITEM|TECHNOLOGY|PROCESS|OTHER",
            "properties": {{"key": "value"}}
        }}
    ],
    "relationships": [
        {{
            "source": "entity name",
            "target": "entity name",
            "type": "relationship type (e.g., PART_OF, TRANSFORMS_INTO, DEPENDS_ON)",
            "properties": {{"key": "value"}}
        }}
    ]
}}

Text to analyze:
{text}

Return only valid JSON, no additional text. Focus on the most important entities and relationships. Limit to 8-10 entities and 6-8 relationships per chunk. Prioritize entities that are central to the content and have clear relationships. Avoid extracting redundant or duplicate entities."""

def extract_knowledge_graph(text: str, source: str = "unknown") -> Dict[str, Any]:
    """
    Extract entities and relationships from text using LLM
    
    Args:
        text: Text content to analyze
        source: Source identifier (e.g., PDF filename or URL)
    
    Returns:
        Dictionary with 'entities' and 'relationships' lists
    """
    if not llm:
        logger.warning("LLM not initialized, skipping knowledge graph extraction")
        return {"entities": [], "relationships": []}
    
    try:
        # Limit text length to avoid token limits
        max_chunk_size = 4000
        if len(text) > max_chunk_size:
            text = text[:max_chunk_size] + "..."
        
        prompt = PromptTemplate.from_template(KG_EXTRACTION_PROMPT)
        parser = JsonOutputParser()
        
        chain = prompt | llm | parser
        
        result = chain.invoke({"text": text})
        
        # Validate and clean the result
        if not isinstance(result, dict):
            logger.warning(f"LLM returned non-dict result: {type(result)}")
            return {"entities": [], "relationships": [], "content_type": "general"}
        
        # Ensure we have the expected structure
        if "entities" not in result:
            result["entities"] = []
        if "relationships" not in result:
            result["relationships"] = []
        if "content_type" not in result:
            result["content_type"] = "general"
        
        # Log content type for debugging
        content_type = result.get("content_type", "general")
        logger.info(f"Detected content type: {content_type}")
        
        # Clean and validate entities
        cleaned_entities = []
        for entity in result.get("entities", []):
            if not isinstance(entity, dict):
                continue
            if "name" not in entity or not entity.get("name", "").strip():
                continue
            if "type" not in entity:
                entity["type"] = "OTHER"
            if "properties" not in entity:
                entity["properties"] = {}
            entity["properties"]["source"] = source
            entity["properties"]["content_type"] = content_type
            cleaned_entities.append(entity)
        
        # Clean and validate relationships
        cleaned_relationships = []
        for rel in result.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            if "source" not in rel or "target" not in rel:
                continue
            if not rel.get("source", "").strip() or not rel.get("target", "").strip():
                continue
            if "type" not in rel:
                rel["type"] = "RELATED_TO"
            if "properties" not in rel:
                rel["properties"] = {}
            rel["properties"]["source"] = source
            rel["properties"]["content_type"] = content_type
            cleaned_relationships.append(rel)
        
        logger.info(f"Extracted {len(cleaned_entities)} entities and {len(cleaned_relationships)} relationships from text (type: {content_type})")
        
        return {
            "entities": cleaned_entities,
            "relationships": cleaned_relationships,
            "content_type": content_type
        }
    except Exception as e:
        logger.error(f"Error extracting knowledge graph: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"entities": [], "relationships": []}

def store_knowledge_graph(
    entities: List[Dict],
    relationships: List[Dict],
    user_id: str,
    source: str,
    task_id: Optional[str] = None,
    driver: Optional[Any] = None
) -> None:
    """
    Store entities and relationships in Neo4j
    
    Args:
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
        user_id: User ID for tagging
        source: Source identifier
        task_id: Optional task ID for isolation
        driver: Optional Neo4j driver (will create if not provided)
    """
    if not entities and not relationships:
        logger.info("No entities or relationships to store")
        return
    
    should_close = False
    if driver is None:
        try:
            driver = get_neo4j_driver()
            should_close = True
        except Exception as e:
            logger.error(f"Cannot connect to Neo4j: {str(e)}")
            return
    
    try:
        with driver.session() as session:
            # Create entities first
            entity_count = 0
            for entity in entities:
                entity_name = entity.get("name", "").strip()
                if not entity_name:
                    continue
                
                entity_type = entity.get("type", "OTHER")
                properties = entity.get("properties", {}).copy()
                properties["user_id"] = user_id
                properties["source"] = source
                properties["type"] = entity_type  # Store type as property too
                if task_id:
                    properties["task_id"] = task_id
                
                # Check for similar existing entities within the same user and task
                # First, try to find exact match with user_id and task_id
                check_query = """
                MATCH (e:Entity {name: $name, user_id: $user_id})
                WHERE ($task_id IS NULL OR e.task_id = $task_id)
                RETURN e.name as name, e.task_id as existing_task_id
                LIMIT 1
                """
                existing = session.run(check_query, name=entity_name, user_id=user_id, task_id=task_id).single()
                
                if existing:
                    # Entity already exists for this user+task, just update properties
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
                    except Exception as e:
                        logger.warning(f"Failed to update entity {entity_name}: {str(e)}")
                
                # Check for similar entities (fuzzy matching) within same user+task
                # Only check if no exact match found
                similarity_query = """
                MATCH (e:Entity {user_id: $user_id})
                WHERE ($task_id IS NULL OR e.task_id = $task_id)
                RETURN e.name as name
                """
                existing_names = [record["name"] for record in session.run(similarity_query, user_id=user_id, task_id=task_id)]
                
                # Check similarity (threshold: 0.90 = 90% similar - stricter to reduce redundancy)
                similar_entity = None
                for existing_name in existing_names:
                    similarity = SequenceMatcher(None, entity_name.lower(), existing_name.lower()).ratio()
                    if similarity >= 0.90:
                        similar_entity = existing_name
                        logger.info(f"Found similar entity: '{entity_name}' similar to '{existing_name}' (similarity: {similarity:.2f})")
                        break
                    
                    # Also check if one is a substring of the other (for abbreviations/variations)
                    if len(entity_name) > 5 and len(existing_name) > 5:
                        if entity_name.lower() in existing_name.lower() or existing_name.lower() in entity_name.lower():
                            # Check if the shorter one is at least 80% of the longer one
                            shorter = min(len(entity_name), len(existing_name))
                            longer = max(len(entity_name), len(existing_name))
                            if shorter / longer >= 0.80:
                                similar_entity = existing_name
                                logger.info(f"Found substring match: '{entity_name}' matches '{existing_name}'")
                                break
                
                if similar_entity:
                    # Use the existing similar entity instead of creating a new one
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
                    except Exception as e:
                        logger.warning(f"Failed to merge similar entity {entity_name} with {similar_entity}: {str(e)}")
                
                # Create new entity - ensure it's isolated by user_id and task_id
                query = f"""
                MERGE (e:Entity {{name: $name, user_id: $user_id, task_id: $task_id}})
                SET e:`{entity_type}`
                SET e += $properties
                RETURN e
                """
                
                try:
                    result = session.run(query, name=entity_name, user_id=user_id, task_id=task_id, properties=properties)
                    result.consume()  # Consume result to ensure query executes
                    entity_count += 1
                except Exception as e:
                    # Fallback: create with Entity label only, store type as property
                    logger.warning(f"Failed to set label {entity_type} for entity {entity_name}, using fallback: {str(e)}")
                    try:
                        fallback_query = """
                        MERGE (e:Entity {name: $name, user_id: $user_id, task_id: $task_id})
                        SET e += $properties
                        RETURN e
                        """
                        result = session.run(fallback_query, name=entity_name, user_id=user_id, task_id=task_id, properties=properties)
                        result.consume()
                        entity_count += 1
                    except Exception as e2:
                        logger.error(f"Failed to create entity {entity_name}: {str(e2)}")
                        continue
            
            # Create relationships - use a single efficient query pattern
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
                
                # Use MERGE to ensure nodes exist (with user_id and task_id), then create relationship
                # This ensures relationships are only created between nodes in the same user+task context
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
                        properties=properties
                    )
                    result.consume()  # Consume result to ensure query executes
                    rel_count += 1
                except Exception as e:
                    logger.warning(f"Failed to create relationship {source_entity} -> {target_entity}: {str(e)}")
                    continue
            
            logger.info(f"Stored {entity_count}/{len(entities)} entities and {rel_count}/{len(relationships)} relationships for source: {source}")
    
    except Exception as e:
        logger.error(f"Error storing knowledge graph: {str(e)}")
    finally:
        if should_close:
            driver.close()

def process_documents_for_kg(
    documents: List[Document],
    user_id: str,
    source: str,
    task_id: Optional[str] = None,
    chunk_size: int = 2000
) -> None:
    """
    Process a list of documents to extract and store knowledge graph
    
    Args:
        documents: List of LangChain Document objects
        user_id: User ID for tagging
        source: Source identifier
        task_id: Optional task ID for isolation
        chunk_size: Size of text chunks to process
    """
    if not documents:
        return
    
    try:
        driver = get_neo4j_driver()
    except Exception as e:
        logger.warning(f"Neo4j not available, skipping knowledge graph creation: {str(e)}")
        return
    
    all_entities = []
    all_relationships = []
    
    try:
        # Process each document chunk
        chunk_count = 0
        for doc in documents:
            text = doc.page_content
            if not text.strip():
                continue
            
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count}/{len(documents)} for KG extraction from {source}")
            
            # Extract knowledge graph from this chunk
            kg_data = extract_knowledge_graph(text, source=source)
            
            entities = kg_data.get("entities", [])
            relationships = kg_data.get("relationships", [])
            
            logger.info(f"Chunk {chunk_count}: Extracted {len(entities)} entities, {len(relationships)} relationships")
            
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        logger.info(f"Total extracted: {len(all_entities)} entities, {len(all_relationships)} relationships from {source}")
        
        # Store all extracted knowledge
        if all_entities or all_relationships:
            logger.info(f"Storing knowledge graph for {source} (task_id: {task_id})...")
            store_knowledge_graph(
                all_entities,
                all_relationships,
                user_id,
                source,
                task_id=task_id,
                driver=driver
            )
            logger.info(f"Knowledge graph storage completed for {source} (task_id: {task_id})")
        else:
            logger.warning(f"No entities or relationships extracted from {source}")
    
    except Exception as e:
        logger.error(f"Error processing documents for knowledge graph: {str(e)}")
    finally:
        driver.close()

def process_text_for_kg(
    text: str,
    user_id: str,
    source: str,
    task_id: Optional[str] = None
) -> None:
    """
    Process text content to extract and store knowledge graph
    
    Args:
        text: Text content to analyze
        user_id: User ID for tagging
        source: Source identifier
        task_id: Optional task ID for isolation
    """
    if not text or not text.strip():
        return
    
    try:
        driver = get_neo4j_driver()
    except Exception as e:
        logger.warning(f"Neo4j not available, skipping knowledge graph creation: {str(e)}")
        return
    
    try:
        # Extract knowledge graph
        kg_data = extract_knowledge_graph(text, source=source)
        
        entities = kg_data.get("entities", [])
        relationships = kg_data.get("relationships", [])
        
        # Store knowledge graph
        if entities or relationships:
            store_knowledge_graph(
                entities,
                relationships,
                user_id,
                source,
                task_id=task_id,
                driver=driver
            )
    
    except Exception as e:
        logger.error(f"Error processing text for knowledge graph: {str(e)}")
    finally:
        driver.close()

def get_user_knowledge_graph(user_id: str, task_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """
    Retrieve knowledge graph data for a user, optionally filtered by task_id
    Returns only well-connected nodes (filters out isolated nodes)
    
    Args:
        user_id: User ID to filter by
        task_id: Optional task ID to filter by (if None, returns all user's graphs)
        limit: Maximum number of nodes to return (default: 30 for better visualization)
    
    Returns:
        Dictionary with 'nodes' and 'links' for graph visualization
    """
    try:
        driver = get_neo4j_driver()
    except Exception as e:
        logger.error(f"Cannot connect to Neo4j: {str(e)}")
        return {"nodes": [], "links": []}
    
    try:
        with driver.session() as session:
            # Get nodes with their connection counts (degree centrality)
            # Only include nodes with at least 2 connections (filters isolated nodes more aggressively)
            if task_id:
                query = """
                MATCH (n)
                WHERE n.user_id = $user_id AND n.task_id = $task_id AND n.name IS NOT NULL
                WITH n, 
                     COUNT { (n)--() } as degree
                WHERE degree >= 1
                RETURN DISTINCT n.name as name, 
                       labels(n) as labels,
                       COALESCE(n.type, head(labels(n))) as type,
                       properties(n) as properties,
                       degree
                ORDER BY degree DESC
                LIMIT $limit
                """
                result = session.run(query, user_id=user_id, task_id=task_id, limit=limit)
            else:
                query = """
                MATCH (n)
                WHERE n.user_id = $user_id AND n.name IS NOT NULL
                WITH n, 
                     COUNT { (n)--() } as degree
                WHERE degree >= 1
                RETURN DISTINCT n.name as name, 
                       labels(n) as labels,
                       COALESCE(n.type, head(labels(n))) as type,
                       properties(n) as properties,
                       degree
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
                
                nodes.append({
                    "id": node_id,
                    "name": name,
                    "type": node_type,
                    "label": name,
                    "properties": properties,
                    "degree": degree
                })
            
            # Get relationships between these nodes (only for selected nodes)
            links = []
            if nodes:
                node_names = [node["name"] for node in nodes]
                if task_id:
                    rel_query = """
                    MATCH (source)-[r]->(target)
                    WHERE source.user_id = $user_id 
                      AND target.user_id = $user_id
                      AND source.task_id = $task_id
                      AND target.task_id = $task_id
                      AND source.name IN $node_names
                      AND target.name IN $node_names
                    RETURN source.name as source, 
                           target.name as target,
                           type(r) as relationship_type,
                           properties(r) as properties
                    """
                    rel_result = session.run(rel_query, user_id=user_id, task_id=task_id, node_names=node_names)
                else:
                    rel_query = """
                    MATCH (source)-[r]->(target)
                    WHERE source.user_id = $user_id 
                      AND target.user_id = $user_id
                      AND source.name IN $node_names
                      AND target.name IN $node_names
                    RETURN source.name as source, 
                           target.name as target,
                           type(r) as relationship_type,
                           properties(r) as properties
                    """
                    rel_result = session.run(rel_query, user_id=user_id, node_names=node_names)
                
                for record in rel_result:
                    source_name = record["source"]
                    target_name = record["target"]
                    rel_type = record["relationship_type"]
                    properties = dict(record["properties"]) if record["properties"] else {}
                    
                    if source_name in node_map and target_name in node_map:
                        links.append({
                            "source": node_map[source_name],
                            "target": node_map[target_name],
                            "type": rel_type,
                            "label": rel_type,
                            "properties": properties
                        })
            
            # Filter out nodes with no connections in the final result
            connected_node_ids = set()
            for link in links:
                connected_node_ids.add(link["source"])
                connected_node_ids.add(link["target"])
            
            # Keep nodes that are connected, or if graph is small, keep all
            if len(nodes) <= 20:
                # For small graphs, keep all nodes
                final_nodes = nodes
            else:
                # For larger graphs, only keep connected nodes
                final_nodes = [node for node in nodes if node["id"] in connected_node_ids]
            
            # Rebuild node_map with final nodes
            final_node_map = {}
            for idx, node in enumerate(final_nodes):
                final_node_map[node["name"]] = idx
                node["id"] = idx
            
            # Update link indices
            final_links = []
            for link in links:
                source_name = nodes[link["source"]]["name"] if link["source"] < len(nodes) else None
                target_name = nodes[link["target"]]["name"] if link["target"] < len(nodes) else None
                
                if source_name in final_node_map and target_name in final_node_map:
                    final_links.append({
                        "source": final_node_map[source_name],
                        "target": final_node_map[target_name],
                        "type": link["type"],
                        "label": link["label"],
                        "properties": link["properties"]
                    })
            
            logger.info(f"Returning {len(final_nodes)} nodes and {len(final_links)} links (filtered from {len(nodes)} nodes)")
            return {"nodes": final_nodes, "links": final_links}
    
    except Exception as e:
        logger.error(f"Error retrieving knowledge graph: {str(e)}")
        return {"nodes": [], "links": []}
    finally:
        driver.close()

def get_user_task_ids(user_id: str) -> List[str]:
    """
    Get list of task_ids that have knowledge graphs for a user
    
    Args:
        user_id: User ID to filter by
    
    Returns:
        List of task_id strings
    """
    try:
        driver = get_neo4j_driver()
    except Exception as e:
        logger.error(f"Cannot connect to Neo4j: {str(e)}")
        return []
    
    try:
        with driver.session() as session:
            # Get distinct task_ids for this user
            query = """
            MATCH (n)
            WHERE n.user_id = $user_id AND n.task_id IS NOT NULL
            RETURN DISTINCT n.task_id as task_id
            ORDER BY task_id DESC
            """
            result = session.run(query, user_id=user_id)
            
            task_ids = [record["task_id"] for record in result]
            logger.info(f"Found {len(task_ids)} task_ids for user {user_id}")
            return task_ids
    
    except Exception as e:
        logger.error(f"Error retrieving task IDs: {str(e)}")
        return []
    finally:
        driver.close()

def get_knowledge_graph_history(user_id: str) -> List[Dict[str, Any]]:
    """
    Get history of knowledge graphs with metadata
    
    Args:
        user_id: User ID to filter by
    
    Returns:
        List of dictionaries with task_id, source, node_count, relationship_count, created_at
    """
    try:
        driver = get_neo4j_driver()
    except Exception as e:
        logger.error(f"Cannot connect to Neo4j: {str(e)}")
        return []
    
    try:
        with driver.session() as session:
            query = """
            MATCH (n)
            WHERE n.user_id = $user_id AND n.task_id IS NOT NULL
            WITH n.task_id as task_id, 
                 collect(DISTINCT n.source) as sources,
                 count(DISTINCT n) as node_count
            MATCH (a)-[r]->(b)
            WHERE a.user_id = $user_id AND a.task_id = task_id
            WITH task_id, sources, node_count, count(DISTINCT r) as rel_count
            RETURN task_id, 
                   sources[0] as source,
                   node_count,
                   rel_count,
                   task_id as created_at
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
                    "created_at": record["created_at"]
                })
            
            return history
    
    except Exception as e:
        logger.error(f"Error retrieving KG history: {str(e)}")
        return []
    finally:
        driver.close()


def extract_llm_text(content) -> str:
    """
    Normalize LangChain/Gemini message content into a plain string.
    Handles string, list of content blocks, and dict responses.
    """
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
                # Gemini/LangChain content block
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)

                # Some providers may use content
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

            else:
                # Handle object-like content blocks
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

def get_node_explanation(node_name: str, user_id: str, task_id: Optional[str] = None) -> str:
    """
    Generate explanation for a node using LLM based on its relationships, connections, and embedding context
    
    Args:
        node_name: Name of the node
        user_id: User ID
        task_id: Optional task ID
    
    Returns:
        Explanation string
    """
    if not llm:
        return "Explanation not available (LLM not initialized)"
    
    # Get embedding context from Qdrant
    embedding_context = ""
    try:
        from services.qdrant_service import query_vectorstore_multi_task, get_task_collections
        
        # Determine which task_ids to query
        task_ids_to_query = []
        if task_id:
            task_ids_to_query = [task_id]
        else:
            # Get all task_ids for this user
            collections = get_task_collections(user_id)
            task_ids_to_query = [coll["task_id"] for coll in collections]
        
        if task_ids_to_query:
            # Query Qdrant for relevant text chunks mentioning this node
            # Use the node name as a query to find semantically similar content
            relevant_docs = query_vectorstore_multi_task(
                question=node_name,
                user_id=user_id,
                task_ids=task_ids_to_query,
                k=5  # Get top 5 most relevant chunks
            )
            
            if relevant_docs:
                # Extract text from relevant documents
                context_texts = []
                for doc in relevant_docs:
                    text = doc.page_content.strip()
                    # Only include chunks that actually mention the node (case-insensitive)
                    if node_name.lower() in text.lower():
                        context_texts.append(text[:500])  # Limit each chunk to 500 chars
                
                if context_texts:
                    embedding_context = "\n\nRelevant context from documents:\n"
                    for i, text in enumerate(context_texts[:3], 1):  # Limit to top 3 chunks
                        embedding_context += f"{i}. {text}\n"
    except Exception as e:
        logger.warning(f"Could not retrieve embedding context: {str(e)}")
        # Continue without embedding context
    
    try:
        driver = get_neo4j_driver()
    except Exception as e:
        logger.error(f"Cannot connect to Neo4j: {str(e)}")
        return "Explanation not available"
    
    try:
        with driver.session() as session:
            # Get node and its relationships with more detail
            if task_id:
                query = """
                MATCH (n {name: $node_name, user_id: $user_id, task_id: $task_id})
                OPTIONAL MATCH (n)-[r1]->(connected)
                OPTIONAL MATCH (connected)-[r2]->(n)
                RETURN n, 
                       collect(DISTINCT {rel: type(r1), target: connected.name, target_type: connected.type}) as outgoing,
                       collect(DISTINCT {rel: type(r2), source: connected.name, source_type: connected.type}) as incoming
                LIMIT 1
                """
                result = session.run(query, node_name=node_name, user_id=user_id, task_id=task_id)
            else:
                query = """
                MATCH (n {name: $node_name, user_id: $user_id})
                OPTIONAL MATCH (n)-[r1]->(connected)
                OPTIONAL MATCH (connected)-[r2]->(n)
                RETURN n, 
                       collect(DISTINCT {rel: type(r1), target: connected.name, target_type: connected.type}) as outgoing,
                       collect(DISTINCT {rel: type(r2), source: connected.name, source_type: connected.type}) as incoming
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
            
            # Build comprehensive context for LLM
            context = f"Entity: {node_name}\nType: {node_type}\nSource: {source}\n"
            
            # Add all outgoing relationships with context
            if outgoing:
                context += "\nOutgoing connections (what this entity relates to):\n"
                for rel in outgoing[:10]:  # Increased limit to 10
                    target_name = rel.get('target', '')
                    target_type = rel.get('target_type', '')
                    rel_type = rel.get('rel', '')
                    if target_name:
                        type_info = f" ({target_type})" if target_type else ""
                        context += f"- {rel_type} -> {target_name}{type_info}\n"
            
            # Add all incoming relationships with context
            if incoming:
                context += "\nIncoming connections (what relates to this entity):\n"
                for rel in incoming[:10]:  # Increased limit to 10
                    source_name = rel.get('source', '')
                    source_type = rel.get('source_type', '')
                    rel_type = rel.get('rel', '')
                    if source_name:
                        type_info = f" ({source_type})" if source_type else ""
                        context += f"- {source_name}{type_info} {rel_type} -> {node_name}\n"
            
            # Combine with embedding context
            full_context = context + embedding_context
            
            # Generate comprehensive explanation
            prompt = f"""You are explaining an entity from a knowledge graph. Provide a comprehensive, context-aware explanation that:
1. Describes what the entity is based on the document context provided
2. Explains its role and significance in the knowledge graph based on its connections
3. Synthesizes information from both the document content and graph relationships
4. Be informative and educational (3-5 sentences)

Entity Information:
{full_context}

Provide a clear, context-rich explanation that combines the document context with the graph structure:"""
            
            response = llm.invoke(prompt).content
            explanation = extract_llm_text(response)
            return explanation
    
    except Exception as e:
        logger.error(f"Error generating node explanation: {str(e)}")
        return f"Error generating explanation: {str(e)}"
    finally:
        driver.close()

def query_kg_for_question(
    question: str,
    user_id: str,
    task_ids: Optional[List[str]] = None,
    max_entities: int = 10
) -> str:
    """
    Query knowledge graph to find relevant entities and relationships for a question
    
    Args:
        question: User's question
        user_id: User ID
        task_ids: Optional list of task_ids to filter by
        max_entities: Maximum number of entities to return
    
    Returns:
        Context string with relevant entities and relationships
    """
    if not llm:
        logger.debug("LLM not initialized, skipping KG query")
        return ""
    
    try:
        driver = get_neo4j_driver()
    except Exception as e:
        logger.error(f"Cannot connect to Neo4j: {str(e)}")
        return ""
    
    try:
        with driver.session() as session:
            # Extract entities from question using LLM
            entity_extraction_prompt = f"""Extract key entities, concepts, or topics from this question that might be in a knowledge graph.
            
Question: {question}

Return a JSON list of 3-5 key terms (entities, concepts, topics) that should be searched in the knowledge graph.
Format: ["term1", "term2", "term3"]

Terms:"""
            
            try:
                response = llm.invoke(entity_extraction_prompt)
                entity_result = extract_llm_text(response.content)
                # Try to parse JSON from response
                import json
                import re
                json_match = re.search(r'\[.*?\]', entity_result, re.DOTALL)
                if json_match:
                    search_terms = json.loads(json_match.group())
                else:
                    # Fallback: extract quoted terms
                    search_terms = re.findall(r'"([^"]+)"', entity_result)
                    if not search_terms:
                        # Last resort: use question words
                        search_terms = [w for w in question.split() if len(w) > 3][:5]
            except Exception as e:
                logger.warning(f"Error extracting entities from question: {str(e)}")
                # Fallback: use question words
                search_terms = [w for w in question.split() if len(w) > 3][:5]
            
            if not search_terms:
                logger.debug("No search terms extracted from question")
                return ""
            
            # Build query to find matching entities
            if task_ids:
                task_filter = "AND n.task_id IN $task_ids"
                connected_task_filter = "AND connected.task_id IN $task_ids"
                params = {"user_id": user_id, "task_ids": task_ids, "search_terms": search_terms}
            else:
                task_filter = ""
                connected_task_filter = ""
                params = {"user_id": user_id, "search_terms": search_terms}
            
            # Find entities matching search terms (fuzzy match)
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
    
    except Exception as e:
        logger.error(f"Error querying KG for question: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ""
    finally:
        driver.close()

def get_relevant_nodes_for_question(
    question: str,
    user_id: str,
    task_ids: Optional[List[str]] = None,
    max_nodes: int = 10
) -> List[Dict[str, Any]]:
    """
    Get relevant nodes from knowledge graph for a question
    
    Args:
        question: User's question
        user_id: User ID
        task_ids: Optional list of task_ids to filter by
        max_nodes: Maximum number of nodes to return
    
    Returns:
        List of node dictionaries with name, type, and connections
    """
    if not llm:
        return []
    
    try:
        driver = get_neo4j_driver()
    except Exception as e:
        logger.error(f"Cannot connect to Neo4j: {str(e)}")
        return []
    
    try:
        with driver.session() as session:
            # Extract entities from question using LLM
            entity_extraction_prompt = f"""Extract key entities, concepts, or topics from this question that might be in a knowledge graph.
            
Question: {question}

Return a JSON list of 3-5 key terms (entities, concepts, topics) that should be searched in the knowledge graph.
Format: ["term1", "term2", "term3"]

Terms:"""
            
            try:
                response = llm.invoke(entity_extraction_prompt)
                entity_result = extract_llm_text(response.content)
                import json
                import re
                json_match = re.search(r'\[.*?\]', entity_result, re.DOTALL)
                if json_match:
                    search_terms = json.loads(json_match.group())
                else:
                    search_terms = re.findall(r'"([^"]+)"', entity_result)
                    if not search_terms:
                        search_terms = [w for w in question.split() if len(w) > 3][:5]
            except Exception as e:
                logger.warning(f"Error extracting entities from question: {str(e)}")
                search_terms = [w for w in question.split() if len(w) > 3][:5]
            
            if not search_terms:
                return []
            
            # Build query to find matching entities
            if task_ids:
                task_filter = "AND n.task_id IN $task_ids"
                params = {"user_id": user_id, "task_ids": task_ids, "search_terms": search_terms}
            else:
                task_filter = ""
                params = {"user_id": user_id, "search_terms": search_terms}
            
            # Find entities matching search terms
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
            RETURN DISTINCT n.name as name,
                   COALESCE(n.type, 'OTHER') as type,
                   properties(n) as properties
            """
            
            params["max_nodes"] = max_nodes
            result = session.run(query, params)
            
            nodes = []
            for record in result:
                nodes.append({
                    "name": record["name"],
                    "type": record.get("type", "OTHER"),
                    "properties": dict(record["properties"]) if record["properties"] else {}
                })
            
            return nodes
    
    except Exception as e:
        logger.error(f"Error getting relevant nodes: {str(e)}")
        return []
    finally:
        driver.close()