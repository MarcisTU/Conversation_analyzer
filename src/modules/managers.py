from src.modules.constants import (
    FS_ACCESS_KEY,
    FS_ENDPOINT,
    FS_SECRET_KEY,
    FS_USE_SECURE,
    MQ_URL,
)
from src.modules.file_storage_client import MinioManager
from src.modules.mq_connection_manager import RabbitMQManager


rmq_manager = RabbitMQManager(url=MQ_URL)

minio_manager = MinioManager(
    endpoint=FS_ENDPOINT,
    access_key=FS_ACCESS_KEY,
    secret_key=FS_SECRET_KEY,
    secure=FS_USE_SECURE,
)