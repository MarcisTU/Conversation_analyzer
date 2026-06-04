from loguru import logger
from miniopy_async import Minio


class MinioManager:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.client = None

    def init_client(self):
        """Initialize the client (Call this during FastAPI startup)"""
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

        logger.info(f"MinIO file storage client successfully initialized at {self.endpoint}")

    async def close_client(self):
        """Close the client (Call this during FastAPI shutdown)"""
        if self.client:
            await self.client.close_session()

    def get_client(self) -> Minio:
        """Dependency provider for FastAPI routes"""
        if not self.client:
            raise RuntimeError("Minio client is not initialized.")
        return self.client
