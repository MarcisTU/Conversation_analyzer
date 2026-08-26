import os
from pathlib import Path


# the projects ./src dir
SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent


# File storage client constants from environment
FS_ENDPOINT=f"{os.environ['FS_HOST']}:{os.environ['FS_PORT']}"
FS_ACCESS_KEY=os.environ["FS_ACCESS_KEY"]
FS_SECRET_KEY=os.environ["FS_SECRET_KEY"]
FS_USE_SECURE=os.environ["FS_USE_SECURE"].lower() == "true"

# Message queue constants
MQ_URL=os.environ["RABBITMQ_URL"]
