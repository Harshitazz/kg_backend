"""
Storage abstraction layer to support multiple backends:
- AWS S3 (current)
- Cloudflare R2 (S3-compatible, cheaper)
- Local filesystem (for small deployments)
"""
import os
import datetime
from abc import ABC, abstractmethod
from typing import Optional
import boto3
from botocore.config import Config


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def upload_file(self, file_path: str, key: str) -> str:
        """Upload file and return URL"""
        pass
    
    @abstractmethod
    def download_file(self, key: str, local_path: str) -> None:
        """Download file from storage"""
        pass
    
    @abstractmethod
    def delete_file(self, key: str) -> None:
        """Delete file from storage"""
        pass
    
    @abstractmethod
    def list_objects(self, prefix: str) -> list:
        """List objects with given prefix"""
        pass
    
    @abstractmethod
    def get_file_url(self, key: str) -> str:
        """Get public URL for file"""
        pass


class S3Storage(StorageBackend):
    """AWS S3 storage backend"""
    
    def __init__(self):
        self.bucket_name = os.getenv("AWS_S3_BUCKET_NAME","vectra-mind")
        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    
    def upload_file(self, file_path: str, key: str) -> str:
        if not self.bucket_name:
            raise ValueError("Bucket name is not configured")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.client.upload_file(file_path, self.bucket_name, key)
        return self.get_file_url(key)
    
    def download_file(self, key: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else ".", exist_ok=True)
        self.client.download_file(self.bucket_name, key, local_path)
    
    def delete_file(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket_name, Key=key)
    
    def list_objects(self, prefix: str) -> list:
        response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        return response.get("Contents", [])
    
    def get_file_url(self, key: str) -> str:
        if not self.bucket_name:
            raise ValueError("Bucket name is not configured")
        region = os.getenv("AWS_REGION", "us-east-1")
        return f"https://{self.bucket_name}.s3.{region}.amazonaws.com/{key}"

def get_storage_backend() -> StorageBackend:
    """Get storage backend based on STORAGE_BACKEND environment variable"""  
    return S3Storage()


storage = get_storage_backend()

def upload_file_to_storage(file_path: str, key: str) -> str:
    """Upload file to configured storage backend"""
    return storage.upload_file(file_path, key)

def download_file_from_storage(key: str, local_path: str) -> None:
    """Download file from configured storage backend"""
    storage.download_file(key, local_path)

def delete_file_from_storage(key: str) -> None:
    """Delete file from configured storage backend"""
    storage.delete_file(key)

def list_storage_objects(prefix: str) -> list:
    """List objects in storage with given prefix"""
    return storage.list_objects(prefix)
