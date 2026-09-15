from typing import Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ConfigDict

from src.models.enums import TaskStatus, FeatureStatus, FeatureName
from src.models.results import Results


class TaskRead(BaseModel):
    # This configuration makes it easy to work with ORM data structures seamlessly
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_uuid: str
    status: TaskStatus = TaskStatus.NOT_SET
    callback_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TaskUpdate(BaseModel):
    # This configuration makes it easy to work with ORM data structures seamlessly
    model_config = ConfigDict(from_attributes=True)

    status: Optional[TaskStatus] = None
    results: Optional[Results] = None


class WorkerStatus(BaseModel):
    worker_id: str
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeatureInTaskRead(BaseModel):
    id: int
    task_id: int
    feature_id: int
    status: FeatureStatus
    result_data: Optional[Results] = None
    name: FeatureName
    order_idx: int

    class Config:
        from_attributes = True
