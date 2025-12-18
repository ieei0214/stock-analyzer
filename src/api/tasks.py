"""
Background Tasks API endpoints.

Provides endpoints for managing background tasks like batch analysis.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Optional
from datetime import datetime
import uuid

from ..database.models import TaskStatus, APIResponse

router = APIRouter()

# In-memory task storage (will be replaced with proper task queue)
# This is a placeholder implementation
active_tasks: Dict[str, TaskStatus] = {}


@router.get("/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a background task."""
    task = active_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/active", response_model=list[TaskStatus])
async def get_active_tasks():
    """Get all active background tasks."""
    return [
        task for task in active_tasks.values()
        if task.status in ("pending", "running")
    ]


@router.post("/{task_id}/cancel", response_model=APIResponse)
async def cancel_task(task_id: str):
    """Cancel a running background task."""
    task = active_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=400,
            detail=f"Task already {task.status}"
        )

    # Update task status
    task.status = "cancelled"
    task.updated_at = datetime.now()
    task.message = "Task cancelled by user"

    return APIResponse(success=True, message="Task cancelled")


def create_task(task_type: str = "analysis") -> TaskStatus:
    """Create a new background task (helper function)."""
    task_id = str(uuid.uuid4())
    task = TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message=f"Starting {task_type}...",
        created_at=datetime.now()
    )
    active_tasks[task_id] = task
    return task


def update_task(task_id: str, **kwargs) -> Optional[TaskStatus]:
    """Update a task's status (helper function)."""
    task = active_tasks.get(task_id)
    if task:
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.updated_at = datetime.now()
    return task


def complete_task(task_id: str, result: Optional[dict] = None) -> Optional[TaskStatus]:
    """Mark a task as completed (helper function)."""
    task = active_tasks.get(task_id)
    if task:
        task.status = "completed"
        task.progress = 100.0
        task.result = result
        task.updated_at = datetime.now()
    return task


def fail_task(task_id: str, error: str) -> Optional[TaskStatus]:
    """Mark a task as failed (helper function)."""
    task = active_tasks.get(task_id)
    if task:
        task.status = "failed"
        task.message = error
        task.updated_at = datetime.now()
    return task
