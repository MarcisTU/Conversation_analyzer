import asyncio
import os

import aio_pika
from loguru import logger


class RabbitMQManager:
    """Manages RabbitMQ connections."""

    def __init__(self, url: str):
        self.url = url
        self.connection = None
        self.channel = None

    async def connect(self, max_retries: int = 5, retry_delay: int = 5):
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Connecting to RabbitMQ (Attempt {attempt}/{max_retries})...")
                # connect_robust handles automatic reconnects
                self.connection = await aio_pika.connect_robust(self.url)
                self.channel = await self.connection.channel()

                await self.channel.declare_queue(os.environ["REQUEST_QUEUE"], durable=True)

                logger.info("RabbitMQ successfully connected and channel initialized.")
                return
            except Exception as e:
                logger.warning(f"RabbitMQ connection failed on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise RuntimeError("Could not connect to RabbitMQ after maximum retries.")
                await asyncio.sleep(retry_delay)

    async def get_channel(self) -> aio_pika.RobustChannel:
        """Ensures the channel is alive before returning it."""
        if not self.channel or self.channel.is_closed:
            logger.warning("RabbitMQ channel was closed. Re-initializing...")
            if not self.connection or self.connection.is_closed:
                await self.connect()
            else:
                self.channel = await self.connection.channel()

        return self.channel

    async def close(self):
        """Gracefully closes connections during shutdown."""
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("RabbitMQ connection closed gracefully.")
