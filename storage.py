"""Backward-compatible storage access layer.

The actual storage implementation lives in the repositories package.
"""

from repositories.storage_repository import (
    S3Storage,
    StorageBackend,
    delete_file_from_storage,
    download_file_from_storage,
    get_storage_backend,
    list_storage_objects,
    storage,
    upload_file_to_storage,
)

__all__ = [
    "StorageBackend",
    "S3Storage",
    "get_storage_backend",
    "storage",
    "upload_file_to_storage",
    "download_file_from_storage",
    "delete_file_from_storage",
    "list_storage_objects",
]

