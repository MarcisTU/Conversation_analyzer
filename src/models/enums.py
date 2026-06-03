from enum import Enum


class SegmentType(str, Enum):
    speech = 'speech'
    noise = 'noise'


class TaskStatus(str, Enum):
    waiting = 'waiting'
    processing = 'processing'
    ready = 'ready'
    failed = 'failed'
    not_set = 'not_set'


class WorkerStatusMessage(str, Enum):
    success = 'success'
    failed = 'failed'
    shutdown = 'shutdown'
    heartbeat = 'heartbeat'
    failed_task = 'failed_task'


class WorkerType(str, Enum):
    llm = 'llm'
    # define more ai worker types here when needed


class LanguageCodes(str, Enum):
    en = 'en'
    lv = 'lv'
    ru = 'ru'
    not_set = 'not_set'


class FeatureStatus(str, Enum):
    WAITING = "waiting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FeatureName(str, Enum):
    DIARIZATION = "diarization"
    STT = "stt"
    EMOTION = "emotion"


class EmotionLabel(str, Enum):
    neutral = 'neutral'
    angry = 'angry'
    disgusted = 'disgusted'
    fearful = 'fearful'
    happy = 'happy'
    other = 'other'
    surprised = 'surprised'
    sad = 'sad'
    unknown = 'unknown'
