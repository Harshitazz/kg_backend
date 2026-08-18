import logging
from typing import Any, Dict, List

from difflib import SequenceMatcher

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

llm = None


def set_llm(llm_instance):
    global llm
    llm = llm_instance


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
    if not llm:
        logging.warning("LLM not initialized, skipping knowledge graph extraction")
        return {"entities": [], "relationships": []}

    try:
        max_chunk_size = 4000
        if len(text) > max_chunk_size:
            text = text[:max_chunk_size] + "..."

        prompt = PromptTemplate.from_template(KG_EXTRACTION_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        result = chain.invoke({"text": text})

        if not isinstance(result, dict):
            logging.warning(f"LLM returned non-dict result: {type(result)}")
            return {"entities": [], "relationships": [], "content_type": "general"}

        if "entities" not in result:
            result["entities"] = []
        if "relationships" not in result:
            result["relationships"] = []
        if "content_type" not in result:
            result["content_type"] = "general"

        content_type = result.get("content_type", "general")
        logging.info(f"Detected content type: {content_type}")

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

        return {
            "entities": cleaned_entities,
            "relationships": cleaned_relationships,
            "content_type": content_type,
        }
    except Exception as exc:
        logging.error(f"Error extracting knowledge graph: {str(exc)}")
        import traceback
        logging.error(traceback.format_exc())
        return {"entities": [], "relationships": []}
