"""
Configuration and environment variable validation
"""
import os
import logging
from typing import Optional
from pydantic import Field, field_validator

logger = logging.getLogger(__name__)


class Settings:
    """Application settings with validation"""
    
    def __init__(self):
        # Required settings
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.clerk_secret_key = os.getenv("CLERK_SECRET_KEY")
        
        # Database settings
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Storage settings
        self.storage_backend = os.getenv("STORAGE_BACKEND", "local")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        # CORS settings
        self.allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
        
        # Server settings
        self.port = int(os.getenv("PORT", "8000"))
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Validate
        self._validate()
    
    def _validate(self):
        """Validate settings"""
        # Validate storage backend
        allowed_backends = ["local", "s3", "r2"]
        if self.storage_backend not in allowed_backends:
            raise ValueError(f"storage_backend must be one of {allowed_backends}, got {self.storage_backend}")
        
        # Validate environment
        allowed_envs = ["development", "staging", "production"]
        if self.environment not in allowed_envs:
            logger.warning(f"Unknown environment: {self.environment}, defaulting to development")
            self.environment = "development"
        
        # Validate storage-specific requirements
        if self.storage_backend == "s3":
            if not self.aws_access_key_id or not self.aws_secret_access_key:
                raise ValueError("AWS credentials required for S3 storage backend")
            if not self.aws_s3_bucket_name:
                raise ValueError("AWS_S3_BUCKET_NAME required for S3 storage backend")


def get_settings() -> Settings:
    """Get application settings with validation"""
    try:
        settings = Settings()  # _validate() is called in __init__
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        # In development, allow minimal settings
        env = os.getenv("ENVIRONMENT", "development")
        if env == "development":
            logger.warning("Using minimal settings for development")
            # Create minimal settings object without validation
            minimal = Settings.__new__(Settings)
            minimal.groq_api_key = os.getenv("GROQ_API_KEY", "")
            minimal.clerk_secret_key = os.getenv("CLERK_SECRET_KEY")
            minimal.storage_backend = "local"
            minimal.environment = "development"
            minimal.allowed_origins = ""
            minimal.port = 8000
            minimal.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            minimal.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            minimal.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
            minimal.qdrant_url = os.getenv("QDRANT_URL")
            minimal.qdrant_api_key = os.getenv("QDRANT_API_KEY")
            minimal.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            minimal.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            minimal.aws_s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
            minimal.aws_region = os.getenv("AWS_REGION", "us-east-1")
            return minimal
        raise
