from typing import Optional

from pydantic import BaseModel


class TaskSubmitRequest(BaseModel):
    user_query: str
    callback_url: Optional[str]
