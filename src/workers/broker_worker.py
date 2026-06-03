import asyncio
import json
import os
import signal
from collections import defaultdict
from datetime import datetime, timezone

import aio_pika
from aio_pika import ExchangeType, IncomingMessage
from loguru import logger

from src.db.service import TaskService
from src.models.enums import TaskStatus, WorkerStatusMessage
from src.models.schemas import TaskUpdate, WorkerStatus
from src.modules.mq_connection_manager import RabbitMQManager

logger.add("./logs/broker_worker.log", rotation="00:00", retention="7 days")


class BrokerWorker:
    def __init__(self, mq_manager):
        self.available_workers: dict[str, dict[str, WorkerStatus]] = defaultdict(dict)
        self.pending_requests: list[dict] = []

        self.mq_manager = mq_manager
        self.exchange = None

        self.request_queue = None
        self.response_queue = None
        self.callback_queue = None

        self.background_print_stats_interval = 5  # in seconds
        self.worker_last_heartbeat_check_interval = 300  # in seconds

    async def connect_message_queues(self):
        await self.mq_manager.channel.set_qos(prefetch_count=1)

        self.request_queue = await self.mq_manager.channel.declare_queue(os.environ["REQUEST_QUEUE"], durable=True)
        self.response_queue = await self.mq_manager.channel.declare_queue(os.environ["RESPONSE_QUEUE"], durable=True)
        self.callback_queue = await self.mq_manager.channel.declare_queue(os.environ["CALLBACK_QUEUE"], durable=True)

        self.exchange = await self.mq_manager.channel.declare_exchange(os.environ["EXCHANGE_NAME"], ExchangeType.DIRECT, durable=True)

        logger.info("RabbitMQ message queues/exchanges declared.")

    async def _print_stats(self):
        try:
            while True:
                await asyncio.sleep(self.background_print_stats_interval)

                available_worker_count = sum(len(workers) for workers in self.available_workers.values())
                logger.info(f"Available workers: {available_worker_count}, Pending requests: {len(self.pending_requests)}")

                if self.pending_requests:
                    oldest = self.pending_requests[0]
                    logger.info(f"Oldest pending request: {oldest['request_id']} | {oldest['queued_at']}")
        except asyncio.CancelledError:
            logger.info("Print stats loop stopped.")

    async def register_worker(self, worker_type: str, worker_id: str):
        is_new = worker_id not in self.available_workers[worker_type]

        self.available_workers[worker_type][worker_id] = WorkerStatus(
            worker_id=worker_id,
            last_heartbeat=datetime.now(timezone.utc)
        )

        if is_new:
            logger.info(f"Worker registered: {worker_type}/{worker_id}")
        else:
            logger.debug(f"Heartbeat updated for: {worker_type}/{worker_id}")

    async def unregister_worker(self, worker_type: str, worker_id: str):
        removed_worker = self.available_workers[worker_type].pop(worker_id, None)

        if removed_worker:
            logger.info(f"Worker removed: {worker_type}/{worker_id}")
        else:
            logger.warning(f"Attempted to unregister non-existent worker: {worker_type}/{worker_id}")

    async def check_available_workers(self):
        for worker_type, workers_dict in self.available_workers.items():
            dead_worker_ids = []

            # Collect workers that haven't responded for more than self.worker_last_heartbeat_check_interval
            for worker_id, status in workers_dict.items():
                time_elapsed = datetime.now(timezone.utc) - status.last_heartbeat

                if time_elapsed.total_seconds() > self.worker_last_heartbeat_check_interval:
                    logger.warning(f"Worker {worker_type}/{worker_id} not responding. Last seen {time_elapsed.total_seconds():.1f}s ago.")
                    dead_worker_ids.append(worker_id)

            for worker_id in dead_worker_ids:
                workers_dict.pop(worker_id, None)

    async def dispatch_request(self, request: dict):
        worker_type = request["worker_type"]

        if not self.available_workers[worker_type]:
            logger.warning(f"No workers available for {worker_type}. Queueing request {request['request_id']}")
            self.pending_requests.append(request)
            return

        routing_key = f"worker.{worker_type}"

        await self.exchange.publish(
            aio_pika.Message(
                body=json.dumps(request).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=routing_key,
        )

        logger.info(f"Dispatched request {request['request_id']} to {worker_type}")

    async def process_pending_requests(self):
        if not self.pending_requests:
            return

        remaining = []
        for request in self.pending_requests:
            worker_type = request["worker_type"]

            if self.available_workers[worker_type]:
                await self.dispatch_request(request)
            else:
                remaining.append(request)

        self.pending_requests = remaining

    async def handle_api_request(self, message: IncomingMessage):
        await self.check_available_workers()

        async with message.process():
            try:
                payload = json.loads(message.body.decode())
                payload["queued_at"] = datetime.utcnow().isoformat()
                logger.info(f"Received API request: {payload['request_id']}")

                await self.dispatch_request(payload)

            except Exception as e:
                logger.exception(e)

    async def handle_worker_response(self, message: IncomingMessage):
        async with message.process():
            try:
                payload = json.loads(message.body.decode())
                # TODO Create payload message pydantic class
                worker_status_message = payload["status"]
                worker_type = payload["worker_type"]
                worker_id = payload["worker_id"]

                await self.register_worker(worker_type, worker_id)

                if worker_status_message == WorkerStatusMessage.shutdown.value:
                    await self.unregister_worker(worker_type, worker_id)
                    return
                elif worker_status_message == WorkerStatusMessage.heartbeat.value:
                    return
                elif worker_status_message == WorkerStatusMessage.failed_task.value:
                    await TaskService.update_task(
                        task_uuid=payload["request_id"],
                        update_data=TaskUpdate(
                            status=TaskStatus.failed,
                        )
                    )
                    return

                ### Received actual task result response from worker..
                task_uuid = payload["request_id"]
                callback_url = payload["callback_url"]
                task_result = payload["result"]

                logger.info(f"Received result response for task {task_uuid}")

                await TaskService.update_task(
                    task_uuid=task_uuid,
                    update_data=TaskUpdate(
                        status=TaskStatus.ready,
                        llm_result_text=task_result
                    )
                )

                if callback_url is not None and len(callback_url.strip()) > 0:
                    # Forward to Callback Worker for sending results back to client
                    callback_payload = {
                        "request_id": task_uuid,
                        "result": task_result,
                        "callback_url": callback_url
                    }

                    await self.mq_manager.channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(callback_payload).encode(),
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        ),
                        routing_key=os.environ["CALLBACK_QUEUE"],
                    )
                    logger.info(f"Result for task {task_uuid} sent to callback worker")

                await self.process_pending_requests()

            except Exception as e:
                logger.exception(e)

    async def requeue_pending_on_exit(self):
        """Moves in-memory pending_requests back to RabbitMQ before exit."""
        if not self.pending_requests:
            return

        logger.info(f"Requeuing {len(self.pending_requests)} pending requests to RabbitMQ...")

        for request in self.pending_requests:
            try:
                await self.mq_manager.channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(request).encode(),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=os.environ["REQUEST_QUEUE"],
                )
            except Exception as e:
                logger.error(f"Failed to requeue request {request.get('request_id')}: {e}")

        self.pending_requests.clear()

    async def run(self):
        print_status_task = asyncio.create_task(self._print_stats())

        request_consumer_tag = await self.request_queue.consume(self.handle_api_request)
        await self.response_queue.consume(self.handle_worker_response)

        logger.info("Broker worker started. Press Ctrl+C to exit.")

        try:
            await asyncio.Future()
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Shutdown signal received...")
        finally:
            # Shutdown & Stop accepting new requests from RabbitMQ first
            if self.request_queue:
                await self.request_queue.cancel(request_consumer_tag)
                logger.info("Stopped consuming new requests.")

            await self.requeue_pending_on_exit()

            print_status_task.cancel()

            logger.info("Broker worker shut down gracefully.")



async def main():
    mq_manager = RabbitMQManager(os.environ["RABBITMQ_URL"])
    await mq_manager.connect()

    broker = BrokerWorker(mq_manager=mq_manager)
    await broker.connect_message_queues()


    loop = asyncio.get_running_loop()
    current_task = asyncio.current_task()


    def handle_exit_signal():
        logger.warning("Received stop signal (SIGTERM/SIGINT). Initiating broker graceful shutdown...")
        if current_task:
            current_task.cancel()


    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig=sig, callback=handle_exit_signal)

    try:
        await broker.run()
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
