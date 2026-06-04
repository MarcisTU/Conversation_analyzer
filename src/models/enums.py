from enum import Enum


class SegmentType(str, Enum):
    speech = 'speech'
    noise = 'noise'


class TaskStatus(str, Enum):
    WAITING = "WAITING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    NOT_SET = 'NOT_SET'


class WorkerStatusMessage(str, Enum):
    success = 'success'
    failed = 'failed'
    shutdown = 'shutdown'
    heartbeat = 'heartbeat'
    failed_task = 'failed_task'


class WorkerType(str, Enum):
    audio_diarization = 'audio_diarization'
    audio_emotions = 'audio_emotions'
    audio_stt = 'audio_stt'


class LanguageCodes(str, Enum):
    en = 'en'
    lv = 'lv'
    ru = 'ru'
    not_set = 'not_set'


class FeatureStatus(str, Enum):
    WAITING = "WAITING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class FeatureName(str, Enum):
    diarization = "diarization"
    stt = "stt"
    emotion = "emotion"


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


class FileBucketNames(str, Enum):
    request_files_unprocessed = 'request_files_unprocessed'
    request_files_processed = 'request_files_processed'
