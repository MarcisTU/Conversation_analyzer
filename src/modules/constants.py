import os
from pathlib import Path


# the projects ./src dir
SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent


# File storage client constants from environment
FS_HOST = os.getenv("FS_HOST", "localhost")
FS_PORT = os.getenv("FS_PORT", "9000")

FS_ENDPOINT = f"{FS_HOST}:{FS_PORT}"
FS_ACCESS_KEY = os.getenv("FS_ACCESS_KEY", "minioadmin")
FS_SECRET_KEY = os.getenv("FS_SECRET_KEY", "minioadmin")
FS_USE_SECURE = os.getenv("FS_USE_SECURE", "false").lower() == "true"

# Message queue constants
MQ_URL = os.getenv(
    "RABBITMQ_URL",
    "amqp://guest:guest@localhost:5672/",
)
