from typing import List

from pydantic import BaseModel, Field

from src.models.enums import LanguageCodes, SegmentType, EmotionLabel


class ResultEmotion(BaseModel):
    label: EmotionLabel = EmotionLabel.unknown
    confidence: float = 0.0


class ResultsSegment(BaseModel):
    start_time: float = Field(default=0.0)
    end_time: float = Field(default=0.0)
    type: SegmentType = Field(default=SegmentType.noise)
    text: str = ""
    emotions: List[ResultEmotion] = Field(default_factory=list)
    user_id: int = 0


class Results(BaseModel):
    segments: List[ResultsSegment] = Field(default_factory=list)
    length_sec: float = 0
    detected_language_code: LanguageCodes = Field(default=LanguageCodes.not_set)
