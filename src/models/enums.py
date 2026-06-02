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
