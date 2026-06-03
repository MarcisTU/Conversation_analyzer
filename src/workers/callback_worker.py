import asyncio
import json
import os
import signal
from datetime import datetime

import httpx
from aio_pika import IncomingMessage
from loguru import logger

from src.modules.mq_connection_manager import RabbitMQManager

os.makedirs("./logs", exist_ok=True)
logger.add("./logs/callback_worker.log", rotation="00:00", retention="7 days")


class CallbackWorker:
    def __init__(self, mq_manager):
        self.mq_manager = mq_manager
        self.callback_queue = None

        self.http_client = httpx.AsyncClient(timeout=10.0)

    async def connect_message_queues(self):
        # Process up to 10 callbacks concurrently per worker instance
        await self.mq_manager.channel.set_qos(prefetch_count=10)

        self.callback_queue = await self.mq_manager.channel.declare_queue(os.environ["CALLBACK_QUEUE"], durable=True)
        logger.info(f"Callback Worker linked and listening on: {os.environ["CALLBACK_QUEUE"]}")

    async def send_callback_with_retry(self, callback_url: str, payload: dict, max_tries: int = 3) -> bool:
        """Sends the HTTP POST callback, retrying up to max_tries with backoff."""
        request_id = payload["request_id"]

        for attempt in range(1, max_tries + 1):
            try:
                logger.info(f"Sending callback for {request_id} to {callback_url} (Attempt {attempt}/{max_tries})")

                response = await self.http_client.post(callback_url, json=payload)

                # Raise exception for 4xx/5xx status codes to trigger retry logic
                response.raise_for_status()

                logger.info(f"Successfully delivered callback for {request_id}. Status: {response.status_code}")
                return True

            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning(f"Attempt {attempt} failed for task {request_id}: {exc}")

                if attempt < max_tries:
                    # Exponential backoff: Wait 2s, then 4s...
                    sleep_time = 2 ** attempt
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached. Callback delivery failed permanently for task {request_id}.")

        return False

    async def process_callback(self, message: IncomingMessage):
        asyncio.create_task(self._safe_execute_callback(message))

    async def _safe_execute_callback(self, message: IncomingMessage):
        async with (message.process(requeue=False)):
            try:
                payload = json.loads(message.body.decode())
                request_id = payload["request_id"]
                callback_url = payload["callback_url"]

                if not callback_url:
                    logger.error(f"Missing callback_url in message payload for task {request_id}. Dropping.")
                    return

                # Universally handle both absolute URLs and relative paths
                final_callback_url = None
                if "localhost" in callback_url or "127.0.0.1" in callback_url:  # locally sent callback test
                    if "localhost" in callback_url:
                        final_callback_url = callback_url.replace("localhost", "host.docker.internal")
                    elif "127.0.0.1" in callback_url:
                        final_callback_url = callback_url.replace("127.0.0.1", "host.docker.internal")
                elif callback_url.startswith("http://") or callback_url.startswith("https://") and \
                    (not "localhost" in callback_url or not "127.0.0.1" in callback_url):
                    # If a full URL is provided (like a third-party webhook) and not local server testing url, use it directly
                    final_callback_url = callback_url

                if final_callback_url is None:
                    logger.error(f"Received callback_url for task {request_id} in message payload is faulty and unusable. Skipping.")
                    return

                # Compile structured payload for client webhook consumers
                client_payload = {
                    "request_id": request_id,
                    "status": "completed",
                    "result": payload["result"],
                    "completed_at": datetime.utcnow().isoformat()
                }

                # Async call out to the endpoint
                success = await self.send_callback_with_retry(final_callback_url, client_payload)

                if not success:
                    logger.error(f"Task {request_id} failed to send callback.")

            except Exception as e:
                logger.exception(f"Fatal crash processing item from callback queue: {e}")

    async def run(self):
        await self.callback_queue.consume(self.process_callback)
        logger.info("Callback Worker engine running. Press Ctrl+C to terminate.")

        try:
            await asyncio.Future()  # Keeps loop alive infinitely
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Shutdown sequence activated...")
        finally:
            await self.http_client.aclose()
            await self.mq_manager.close()

            logger.info("Callback Worker disconnected safely.")


async def main():
    mq_manager = RabbitMQManager(os.environ["RABBITMQ_URL"])
    await mq_manager.connect()

    callback_worker = CallbackWorker(mq_manager=mq_manager)
    await callback_worker.connect_message_queues()


    loop = asyncio.get_running_loop()
    current_task = asyncio.current_task()


    def handle_exit_signal():
        logger.warning("Received stop signal (SIGTERM/SIGINT). Initiating broker graceful shutdown...")
        if current_task:
            current_task.cancel()


    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig=sig, callback=handle_exit_signal)


    try:
        await callback_worker.run()
    except asyncio.CancelledError:
        logger.info("Main broker task cancelled via signal.")
    finally:
        logger.info("Cleaning up resources...")
        await mq_manager.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Broker process completely stopped.")

