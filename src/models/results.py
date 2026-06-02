from typing import List

from pydantic import BaseModel, Field

from src.models.enums import LanguageCodes, SegmentType


class ResultsSegment(BaseModel):
    start_time: float = Field(default=0.0)
    end_time: float = Field(default=0.0)
    type: SegmentType = Field(default=SegmentType.noise)
    transcript: str = ""
    emotion: str = ""
    user_id: int = Field(default_factory=int)
    emotions: List[str] = Field(default_factory=list)


class Results(BaseModel):
    segments: List[ResultsSegment] = Field(default=list)
    length_sec: float = 0
    detected_language_code: LanguageCodes = Field(default=LanguageCodes.not_set)
