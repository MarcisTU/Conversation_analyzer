import asyncio
import json
import os
import signal
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import List

import aio_pika
from aio_pika import ExchangeType, IncomingMessage
from loguru import logger
from pydantic import ValidationError

from src.db.service import TaskService
from src.models.enums import TaskStatus, WorkerStatusMessage, FeatureStatus, FeatureName
from src.models.requests import TaskPayload, WorkerResponsePayload
from src.models.results import Results
from src.models.schemas import TaskUpdate, WorkerStatus, FeatureInTaskRead
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

        self.background_print_stats_interval = 5
        self.worker_last_heartbeat_check_interval = 100

    async def connect_message_queues(self):
        await self.mq_manager.channel.set_qos(prefetch_count=1)

        self.request_queue = await self.mq_manager.channel.declare_queue(os.environ["REQUEST_QUEUE"], durable=True)
        self.response_queue = await self.mq_manager.channel.declare_queue(os.environ["RESPONSE_QUEUE"], durable=True)
        self.callback_queue = await self.mq_manager.channel.declare_queue(os.environ["CALLBACK_QUEUE"], durable=True)

        self.exchange = await self.mq_manager.channel.declare_exchange(
            os.environ["EXCHANGE_NAME"], ExchangeType.DIRECT, durable=True
        )

        logger.info("RabbitMQ message queues/exchanges declared.")

    async def _print_stats(self):
        try:
            while True:
                await asyncio.sleep(self.background_print_stats_interval)

                available_worker_count = sum(len(w) for w in self.available_workers.values())
                logger.info(f"Available workers: {available_worker_count}, Pending requests: {len(self.pending_requests)}")
                if self.pending_requests:
                    oldest = self.pending_requests[0]
                    logger.info(f"Oldest pending request: {oldest['request_id']} | {oldest['queued_at']}")

                self.check_available_workers()
        except asyncio.CancelledError:
            logger.info("Print stats loop stopped.")

    def register_worker(self, worker_type: FeatureName, worker_id: str):
        is_new = worker_id not in self.available_workers[worker_type.value]

        self.available_workers[worker_type.value][worker_id] = WorkerStatus(
            worker_id=worker_id,
            last_heartbeat=datetime.now(timezone.utc)
        )
        if is_new:
            logger.info(f"Worker registered: {worker_type.value}.{worker_id}")
        else:
            logger.debug(f"Heartbeat updated for: {worker_type.value}.{worker_id}")

    def unregister_worker(self, worker_type: FeatureName, worker_id: str):
        removed = self.available_workers[worker_type.value].pop(worker_id, None)
        if removed:
            logger.info(f"Worker removed: {worker_type.value}.{worker_id}")
        else:
            logger.warning(f"Attempted to unregister non-existent worker: {worker_type.value}.{worker_id}")

    def check_available_workers(self):
        for worker_type, workers_dict in self.available_workers.items():
            dead = [
                wid for wid, status in workers_dict.items()
                if (datetime.now(timezone.utc) - status.last_heartbeat).total_seconds()
                > self.worker_last_heartbeat_check_interval
            ]
            for wid in dead:
                logger.warning(f"Worker {worker_type}.{wid} timed out, removing.")
                workers_dict.pop(wid, None)

    async def dispatch_feature(self, request_id: str, worker_type: FeatureName, callback_url: str | None, features_in_task_id: int, existing_results: Results):
        """Publish a single feature job to the appropriate worker queue."""
        payload = {
            "request_id": request_id,
            "worker_type": worker_type.value,
            "callback_url": callback_url,
            "queued_at": datetime.utcnow().isoformat(),
            "features_in_task_id": features_in_task_id,
            "existing_results": (
                existing_results.model_dump(mode="json") 
                if existing_results 
                else None
            )
        }

        is_dispatched = False
        if not self.available_workers[worker_type.value]:
            logger.warning(f"No workers for {worker_type.value}. Leaving feature {features_in_task_id} as WAITING.")

            self.pending_requests.append(payload)
        else:
            # currently select random. TODO balanced?
            target_worker_id = random.choice(self.available_workers[worker_type.value])
            routing_key = f"worker.{worker_type.value}.{target_worker_id}"

            await self.exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=routing_key,
            )
            logger.info(f"Dispatched feature '{worker_type.value}' for task {request_id} and worker {routing_key}")
            is_dispatched = True

        return is_dispatched

    async def dispatch_next_feature(self, task_uuid: str):
        """
        Load the task's feature list ordered by order_idx, find the first
        WAITING feature and dispatch it.
        If all features are done, mark the task complete (and send callback).
        """
        task = await TaskService.get_task(task_uuid=task_uuid)
        if task is None:
            logger.error(f"dispatch_next_feature: task {task_uuid} not found")
            return

        # features_in_task are ordered by feature.order_idx ascending
        features: List[FeatureInTaskRead] = await TaskService.get_features_for_task(
            task_uuid=task.task_uuid
        )

        prior_results = [
            f.result_data
            for f in features
            if f.status == FeatureStatus.READY
        ]
        existing_results = (
            prior_results[-1]
            if prior_results
            else None
        )

        next_feature = next(
            (
                f 
                for f in features 
                if f.status == FeatureStatus.WAITING
            ),
            None
        )
        if next_feature is None:
            if any(f.status == FeatureStatus.FAILED for f in features):
                logger.error(f"Task {task_uuid} completed with failures.")

                await TaskService.update_task(
                    task_uuid=task_uuid,
                    update_data=TaskUpdate(
                        status=TaskStatus.FAILED
                    )
                )
            else:
                logger.info(f"All features done for task {task_uuid}. Marking ready.")
                # Get last feature result since that contains all previous and current results
                await TaskService.update_task(
                    task_uuid=task_uuid,
                    update_data=TaskUpdate(
                        status=TaskStatus.READY,
                        results=existing_results  # add latest results as final task result
                    )
                )
                await self._send_callback(task_uuid, existing_results, task.callback_url)
        else:
            is_dispatched = await self.dispatch_feature(
                request_id=task_uuid,
                worker_type=next_feature.name,
                callback_url=task.callback_url,
                features_in_task_id=next_feature.id,
                existing_results=existing_results,
            )

            # only mark as PROCESSING if worker was available and task is dispatched
            if is_dispatched:
                await TaskService.update_feature_status_by_type(
                    task_uuid=task_uuid,
                    feature_name=next_feature.name,
                    new_status=FeatureStatus.PROCESSING,
                )

    async def handle_api_request(self, message: IncomingMessage):
        self.check_available_workers()

        async with message.process():
            try:
                try:
                    payload = TaskPayload.model_validate_json(message.body.decode())
                except ValidationError as e:
                    logger.error(f"Invalid payload format received: {e}")
                    return

                logger.info(f"Received API request for task {payload.task_uuid}")

                await TaskService.update_task(
                    task_uuid=payload.task_uuid,
                    update_data=TaskUpdate(
                        status=TaskStatus.PROCESSING
                    )
                )

                await self.dispatch_next_feature(payload.task_uuid)

            except Exception as e:
                logger.exception(e)

    async def handle_worker_response(self, message: IncomingMessage):
        async with message.process():
            try:
                try:
                    worker_response_payload = WorkerResponsePayload.model_validate_json(message.body.decode())
                except ValidationError as e:
                    logger.error(f"Invalid payload format received: {e}")
                    return

                if worker_response_payload.status == WorkerStatusMessage.startup.value:
                    self.register_worker(worker_response_payload.worker_type, worker_response_payload.worker_id)

                    await self.process_worker_pending_requests(
                        worker_type=worker_response_payload.worker_type,
                        worker_id=worker_response_payload.worker_id
                    )
                elif worker_response_payload.status == WorkerStatusMessage.shutdown.value:
                    self.unregister_worker(worker_response_payload.worker_type, worker_response_payload.worker_id)
                elif worker_response_payload.status == WorkerStatusMessage.heartbeat.value:
                    self.register_worker(worker_response_payload.worker_type, worker_response_payload.worker_id)
                elif worker_response_payload.status == WorkerStatusMessage.failed.value:
                    updated = await TaskService.update_feature_status_by_type(
                        task_uuid=worker_response_payload.task_uuid,
                        feature_name=worker_response_payload.worker_type,
                        new_status=FeatureStatus.FAILED
                    )
                    if not updated:
                        logger.warning(f"Could not find feature entry for task {worker_response_payload.task_uuid} with type {worker_response_payload.worker_type}")

                    await TaskService.update_task(
                        task_uuid=worker_response_payload.task_uuid,
                        update_data=TaskUpdate(
                            status=TaskStatus.FAILED
                        )
                    )

                    logger.error(f"Feature '{worker_response_payload.worker_type.value}' failed for task {worker_response_payload.task_uuid}. Task aborted.")
                else:
                    logger.info(f"Feature '{worker_response_payload.worker_type.value}' completed for task {worker_response_payload.task_uuid}")

                    await TaskService.update_feature_status_by_type(
                        task_uuid=worker_response_payload.task_uuid,
                        feature_name=worker_response_payload.worker_type,
                        new_status=FeatureStatus.READY,
                        result_data=worker_response_payload.result
                    )

                    # assign any pending feature for this specific worker_type
                    await self.process_worker_pending_requests(
                        worker_type=worker_response_payload.worker_type, 
                        worker_id=worker_response_payload.worker_id
                    )

                    # Kick off the next feature for task in the pipeline (or finalize the task)
                    await self.dispatch_next_feature(worker_response_payload.task_uuid)
                    

            except Exception as e:
                logger.exception(e)

    async def _send_callback(self, task_uuid: str, result: Results, callback_url: str | None):
        if not callback_url or not callback_url.strip():
            return

        callback_payload = {
            "request_id": task_uuid,
            "result": result,
            "callback_url": callback_url,
        }
        await self.mq_manager.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(callback_payload).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=os.environ["CALLBACK_QUEUE"],
        )
        logger.info(f"Callback queued for task {task_uuid}")

    async def process_worker_pending_requests(
            self, 
            worker_type: FeatureName, 
            worker_id: str
    ) -> bool:
        if not self.pending_requests:
            return

        remaining = []
        is_dispatched = False
        for request in self.pending_requests:
            req_worker_type = request["worker_type"]

            if self.available_workers[req_worker_type] and req_worker_type == worker_type and not is_dispatched:
                routing_key = f"worker.{req_worker_type}.{worker_id}"

                await self.exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(request).encode(),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=routing_key,
                )

                await TaskService.update_feature_status_by_type(
                    task_uuid=request["request_id"],
                    feature_name=request["worker_type"],
                    new_status=FeatureStatus.PROCESSING,
                )

                logger.info(f"Dispatched pending feature job for task {request['request_id']} to worker with id: {worker_id}")
                is_dispatched = True
            else:
                remaining.append(request)

        self.pending_requests = remaining

    async def run(self):
        # Recover features that were still WAITING when the broker was previously stopped/crashed.
        self.pending_requests = await TaskService.get_pending_feature_requests()

        pending_len = len(self.pending_requests)
        if pending_len > 0:
            logger.info(f"Loaded {pending_len} pending task features to process.")

        print_status_task = asyncio.create_task(self._print_stats())
        request_consumer_tag = await self.request_queue.consume(self.handle_api_request)
        await self.response_queue.consume(self.handle_worker_response)

        logger.info("Broker worker started.")

        try:
            await asyncio.Future()
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Shutdown signal received...")
        finally:
            if self.request_queue:
                await self.request_queue.cancel(request_consumer_tag)
                logger.info("Stopped consuming new requests.")

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
        logger.warning("Received stop signal. Initiating graceful shutdown...")
        if current_task:
            current_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig=sig, callback=handle_exit_signal)

    try:
        await broker.run()
    except asyncio.CancelledError:
        logger.info("Main broker task cancelled.")
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
