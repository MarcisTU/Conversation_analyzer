import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

import aio_pika
from aio_pika import DeliveryMode, Message
from fastapi import FastAPI, HTTPException, Depends, status, Query
from loguru import logger

from src.db.service import TaskService
from src.models.enums import TaskStatus, WorkerType
from src.models.requests import TaskSubmitRequest
from src.models.response import TaskSubmitResponse, TaskStatusResponse
from src.models.schemas import TaskRead
from src.modules.exception_handlers import register_exception_handlers
from src.modules.mq_connection_manager import RabbitMQManager


rmq_manager = RabbitMQManager(os.environ["RABBITMQ_URL"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    await rmq_manager.connect()
    yield
    await rmq_manager.close()


app = FastAPI(lifespan=lifespan)
register_exception_handlers(app)


@app.get("/")
async def root():
    return {"message": "I am alive and well. Thank you for checking in!"}


@app.post(
    path="/api/v1/task_submit",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TaskSubmitResponse
)
async def create_product_search_task(
    payload: TaskSubmitRequest,
    channel: aio_pika.RobustChannel = Depends(rmq_manager.get_channel)
):
    task_uuid = str(uuid.uuid4())

    # TODO Create pydantic model for payload
    task_payload = {
        "request_id": task_uuid,
        "worker_type": WorkerType.llm.value,
        "user_query": payload.user_query,
        "callback_url": payload.callback_url,
        "metadata": {"generated_by": "fastapi_v1_products"}
    }

    try:
        await channel.default_exchange.publish(
            Message(
                body=json.dumps(task_payload).encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=os.environ["REQUEST_QUEUE"],
        )
        logger.info(f"Successfully published task {task_uuid} to {os.environ["REQUEST_QUEUE"]}")

        await TaskService.insert_task(
            user_query=payload.user_query,
            callback_url=payload.callback_url,
            task_uuid=task_uuid
        )

    except Exception as e:
        logger.error(f"Database insertion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize task."
        )

    return TaskSubmitResponse(
        status=TaskStatus.waiting,
        task_uuid=task_uuid,
        message="Task has been queued for processing."
    )

@app.get(
    path="/api/v1/task_status",
    status_code=status.HTTP_200_OK,
    response_model=TaskStatusResponse
)
async def task_status(
    task_uuid: Annotated[
        UUID,
        Query(description="Valid uuid value that is returned when you submit a task to /api/v1/task_submit")
    ] = None
):
    task_uuid_value = str(task_uuid)
    task_data: TaskRead = await TaskService.get_task(task_uuid_value)

    if task_data is not None:
        logger.info(f"Successfully fetched task {task_uuid_value}")
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with task_uuid={task_uuid_value} doesn't exist."
        )

    return TaskStatusResponse(
        status=task_data.status,
        task_uuid=task_uuid_value,
        result_text=task_data.llm_result_text
    )

