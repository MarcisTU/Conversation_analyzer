import argparse
import asyncio
import json
import os
import signal
import uuid

from aio_pika import IncomingMessage, ExchangeType, DeliveryMode, Message
from loguru import logger

from src.models.enums import WorkerStatusMessage, FeatureName
from src.models.requests import WorkerResponsePayload
from src.modules.mq_connection_manager import RabbitMQManager
from src.services.diarization_service import DiarizationService


WORKER_ID = f"worker-{uuid.uuid4().hex[:8]}"

logger.add(f"./logs/worker_{WORKER_ID}.log", rotation="00:00", retention="7 days")


class AudioDiarizationWorker:
    def __init__(self, mq_manager, audio_diarization_service):
        self.mq_manager = mq_manager

        self.queue = None
        self.exchange = None
        self.routing_key = f"worker.{FeatureName.diarization}"

        self.audio_diarization_service = audio_diarization_service

    async def connect_message_queues(self):
        await self.mq_manager.channel.set_qos(prefetch_count=1)

        self.exchange = await self.mq_manager.channel.declare_exchange(
            os.environ["EXCHANGE_NAME"], ExchangeType.DIRECT, durable=True
        )

        # Declare a private queue for this worker and bind it to the exchange
        # We use a non-durable, auto-delete queue so it vanishes when the worker dies
        self.queue = await self.mq_manager.channel.declare_queue(
            f"queue.{WORKER_ID}", auto_delete=True, exclusive=True
        )
        await self.queue.bind(self.exchange, routing_key=self.routing_key)

        logger.info(f"Worker {WORKER_ID} connected and bound to {self.routing_key}")

    async def send_status(self, worker_response: WorkerResponsePayload):
        try:
            await self.mq_manager.channel.default_exchange.publish(
                Message(
                    body=worker_response.model_dump_json().encode(),
                    delivery_mode=DeliveryMode.PERSISTENT,
                ),
                routing_key=os.environ["RESPONSE_QUEUE"],
            )
        except Exception as e:
            logger.error(f"Failed to send status {worker_response.status}: {e}")

    async def heartbeat_loop(self):
        try:
            while True:
                await self.send_status(WorkerResponsePayload(status=WorkerStatusMessage.heartbeat))
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            logger.info("Heartbeat loop stopped.")

    async def process_task(self, message: IncomingMessage):
        async with message.process():
            try:
                payload = json.loads(message.body.decode())
                task_uuid = payload["task_uuid"]

                logger.info(f"Processing task {task_uuid}...")

                result = await self.audio_diarization_service.inference(file_path="")   # TODO fetch file path from minio and pass temporary file handle
                logger.info(f"Finished task {task_uuid}. \nResult: {result}")

                worker_response_data = WorkerResponsePayload(
                    task_uuid=task_uuid,
                    status=WorkerStatusMessage.success,
                    worker_id=WORKER_ID,
                    worker_type=FeatureName.diarization,
                    result=result,
                    callback_url=payload["callback_url"]
                )

                # Send result back (this also acts as a registration for the next task)
                await self.send_status(worker_response_data)
                logger.info(f"Task {task_uuid} completed and result sent.")

            except Exception as e:
                logger.exception(f"Error processing task: {e}")
                await self.send_status(WorkerResponsePayload(status=WorkerStatusMessage.failed_task))

    async def run(self):
        heartbeat_task = asyncio.create_task(self.heartbeat_loop())

        logger.info(f"Worker {WORKER_ID} waiting for tasks...")

        try:
            await self.queue.consume(self.process_task)
            await asyncio.Future()  # Run forever
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Worker shutting down...")
        finally:
            await self.send_status(WorkerResponsePayload(status=WorkerStatusMessage.shutdown))

            heartbeat_task.cancel()


def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Worker args")

    return parser.parse_args()


async def main():
    args = parse_arguments()

    mq_manager = RabbitMQManager(os.environ["RABBITMQ_URL"])
    await mq_manager.connect()

    audio_diarization_service = DiarizationService(args)

    audio_diarization_worker = AudioDiarizationWorker(
        mq_manager=mq_manager,
        audio_diarization_service=audio_diarization_service
    )
    await audio_diarization_worker.connect_message_queues()


    loop = asyncio.get_running_loop()
    current_task = asyncio.current_task()


    def handle_exit_signal():
        logger.warning("Received stop signal (SIGTERM/SIGINT). Initiating broker graceful shutdown...")
        if current_task:
            current_task.cancel()


    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig=sig, callback=handle_exit_signal)

    try:
        await audio_diarization_worker.run()
    except asyncio.CancelledError:
        logger.info("Main worker task cancelled via signal.")
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
