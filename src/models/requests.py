from typing import Optional

from pydantic import BaseModel, Field

from src.models.enums import FileBucketNames, WorkerStatusMessage, FeatureName
from src.models.results import Results


class TaskPayload(BaseModel):
    task_uuid: str
    file_name: str
    bucket_name: FileBucketNames
    callback_url: Optional[str] = None


class WorkerResponsePayload(BaseModel):
    task_uuid: Optional[str] = ""
    status: WorkerStatusMessage
    worker_id: Optional[str] = ""
    worker_type: Optional[FeatureName] = ""
    result: Optional[Results] = Field(default_factory=Results)
    callback_url: Optional[str] = ""
