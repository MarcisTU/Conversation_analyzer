import os
from pathlib import Path

from loguru import logger
from sqlalchemy.engine import URL

# the projects ./src dir
SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent


# File storage client constants from environment
FS_HOST = os.getenv("FS_HOST")
FS_PORT = os.getenv("FS_PORT")

FS_ENDPOINT = f"{FS_HOST}:{FS_PORT}"
FS_ACCESS_KEY = os.getenv("FS_ACCESS_KEY")
FS_SECRET_KEY = os.getenv("FS_SECRET_KEY")
FS_USE_SECURE = os.getenv("FS_USE_SECURE", "false").lower() == "true"

# Message queue constants
MQ_URL = os.getenv("RABBITMQ_URL")

# Database URL setup (use sqlalchemy URL to avoid special characters in passwords and allow more robust creation)
DATABASE_URL = URL.create(
    drivername="postgresql+asyncpg",
    username=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.environ["DB_HOST"],
    port=int(os.environ["DB_PORT"]),
    database=os.environ["DB_NAME"],
)  # <drivername>://<DB_USER>:<DB_PASSWORD>@<DB_HOST>:<DB_PORT>/<DB_NAME>
