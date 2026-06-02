from pydantic import BaseModel, Field


class AiSADSegment(BaseModel):
    start_sec: float = Field(default=0)
    end_sec: float = Field(default=0)
    speaker_id: int = Field(default=-1)
