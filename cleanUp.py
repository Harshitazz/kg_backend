"""
Cleanup service for removing old files from storage
"""
from storage import list_storage_objects, delete_file_from_storage
import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
import logging

logger = logging.getLogger(__name__)

def cleanup_old_files():
    """Deletes files older than 24 hours from storage"""
    try:
        objects = list_storage_objects("")
        deleted_count = 0
        
        for obj in objects:
            last_modified = obj.get("LastModified")
            if isinstance(last_modified, str):
                last_modified = datetime.datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
            elif not isinstance(last_modified, datetime.datetime):
                continue
                
            age_seconds = (datetime.datetime.now(datetime.timezone.utc) - last_modified).total_seconds()
            if age_seconds > 86400:  # 24 hours
                try:
                    delete_file_from_storage(obj["Key"])
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {obj['Key']}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old files from storage")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_loop())  
    try:
        yield
    finally:
        # Cancel cleanup task on shutdown
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # Expected when canceling the task
            pass

async def cleanup_loop():
    """Background task to periodically clean up old files"""
    try:
        while True:
            try:
                cleanup_old_files()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            
            try:
                await asyncio.sleep(86400)  # Run every 24 hours
            except asyncio.CancelledError:
                break
    except asyncio.CancelledError:
        logger.info("Cleanup loop cancelled")
        pass
