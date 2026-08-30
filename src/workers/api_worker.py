import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

import aio_pika
from aio_pika import DeliveryMode, Message
from fastapi import FastAPI, HTTPException, Depends, status, Query, UploadFile

from loguru import logger
from miniopy_async import Minio

from src.db.service import TaskService
from src.models.enums import TaskStatus, FileBucketNames, FeatureName
from src.models.requests import TaskPayload
from src.models.response import TaskSubmitResponse, TaskStatusResponse
from src.models.results import Results
from src.models.schemas import TaskRead
from src.modules.exception_handlers import register_exception_handlers
from src.modules.file_storage_client import MinioManager
from src.modules.mq_connection_manager import RabbitMQManager
from src.modules.security import verify_api_key
from src.modules.constants import FS_ENDPOINT, FS_ACCESS_KEY, FS_SECRET_KEY, FS_USE_SECURE, MQ_URL


rmq_manager = RabbitMQManager(
    url=MQ_URL
)
minio_manager = MinioManager(
    endpoint=FS_ENDPOINT, 
    access_key=FS_ACCESS_KEY,
    secret_key=FS_SECRET_KEY, 
    secure=FS_USE_SECURE
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await rmq_manager.connect()
    minio_manager.init_client()
    yield
    await rmq_manager.close()
    await minio_manager.close_client()


# Protect all routes. If need some to be public can create separately Dependency for each
app = FastAPI(lifespan=lifespan, dependencies=[Depends(verify_api_key)])
register_exception_handlers(app)


@app.get("/")
async def root():
    return {"message": "I am alive and well. Thank you for checking in!"}


@app.post(
    path="/api/v1/task_submit",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TaskSubmitResponse
)
async def create_audio_task(
    file: UploadFile,
    callback_url: str | None = Query(None, description="Callback url for receiving task completion results"),
    file_client: Minio = Depends(minio_manager.get_client),
    channel: aio_pika.RobustChannel = Depends(rmq_manager.get_channel)
):
    task_uuid = str(uuid.uuid4())

    if not await file_client.bucket_exists(FileBucketNames.request_files_unprocessed):
        await file_client.make_bucket(FileBucketNames.request_files_unprocessed)

    await file_client.put_object(
        bucket_name=FileBucketNames.request_files_unprocessed,
        object_name=task_uuid,
        data=file.file,
        length=file.size
    )

    await TaskService.insert_task_with_features(
        callback_url=callback_url,
        task_uuid=task_uuid,
        # features_to_process=[FeatureName.diarization, FeatureName.emotion, FeatureName.stt]
        features_to_process=[FeatureName.diarization]
    )

    task_payload = TaskPayload(
        task_uuid=task_uuid,
        file_name=task_uuid,
        bucket_name=FileBucketNames.request_files_unprocessed,
        callback_url=callback_url
    )

    try:
        await channel.default_exchange.publish(
            Message(
                body=task_payload.model_dump_json().encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=os.environ["REQUEST_QUEUE"],
        )
        logger.info(f"Successfully published task {task_uuid} to {os.environ['REQUEST_QUEUE']}")

    except Exception as e:
        logger.error(f"Database insertion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize task."
        )

    return TaskSubmitResponse(
        status=TaskStatus.WAITING,
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

    # TODO get task result from FeaturesInTask table

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
        result=Results()
    )


@app.post(
    path="/api/v1/delete_task_audio_file",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_task_audio_file(
    task_uuid: Annotated[
        UUID,
        Query(description="Valid uuid value that is returned when you submit a task to /api/v1/task_submit")
    ] = None,
    file_client: Minio = Depends(minio_manager.get_client)
):
    try:
        await file_client.remove_object(FileBucketNames.request_files_unprocessed, str(task_uuid))

    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to delete task audio file task_uuid={task_uuid}. File doesn't exist in the storage."
        )
