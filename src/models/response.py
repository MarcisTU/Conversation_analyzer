from pydantic import BaseModel, Field

from src.models.enums import TaskStatus
from src.models.results import Results


class TaskSubmitResponse(BaseModel):
    status: TaskStatus = Field(default=TaskStatus.waiting)
    task_uuid: str
    message: str


class TaskStatusResponse(BaseModel):
    status: TaskStatus = Field(default=TaskStatus.waiting)
    task_uuid: str
    results: Results = Field(default_factory=Results)
