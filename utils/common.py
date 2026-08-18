import os
from typing import Iterable, List, Optional


def is_valid_task_id(task_id: Optional[str]) -> bool:
    """Validate a task id used across Qdrant and KG features."""
    return bool(task_id and len(task_id) > 20 and "-" in task_id)


def normalize_task_ids(task_ids: Optional[Iterable[str]]) -> List[str]:
    """Filter out invalid task ids while preserving the valid ones."""
    if not task_ids:
        return []
    return [task_id for task_id in task_ids if is_valid_task_id(task_id)]


def format_document_source(source: Optional[str]) -> str:
    """Return a readable source name for document output."""
    if not source or source == "unknown":
        return "document"
    return os.path.basename(source)
