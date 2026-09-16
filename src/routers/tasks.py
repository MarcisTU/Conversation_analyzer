from typing import Annotated
import os
import uuid
from typing import Annotated
from uuid import UUID

import aio_pika
from aio_pika import DeliveryMode, Message
from fastapi import APIRouter, HTTPException, Depends, status, Query, UploadFile
from loguru import logger
from miniopy_async import Minio

from src.db.service import TaskService
from src.models.enums import TaskStatus, FileBucketNames, FeatureName
from src.models.requests import TaskPayload
from src.models.response import TaskSubmitResponse, TaskStatusResponse
from src.models.schemas import TaskRead, TaskResultRead
from src.modules.managers import minio_manager, rmq_manager


router = APIRouter(
    prefix="/api/v1/tasks", 
    tags=["Tasks"],
)


@router.post(
    path="/submit",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TaskSubmitResponse
)
async def create_audio_task(
    file: UploadFile,
    callback_url: str | None = Query(
        None, description="Callback url for receiving task completion results"
    ),
    features_to_process: list[FeatureName] = Query(
        default=[FeatureName.diarization],
        description="Features to process.",
    ),
    file_client: Minio = Depends(minio_manager.get_client),
    channel: aio_pika.RobustChannel = Depends(rmq_manager.get_channel)
):
    task_uuid = str(uuid.uuid4())

    if not await file_client.bucket_exists(FileBucketNames.request_files_unprocessed.value):
        await file_client.make_bucket(FileBucketNames.request_files_unprocessed.value)

    await file_client.put_object(
        bucket_name=FileBucketNames.request_files_unprocessed.value,
        object_name=task_uuid,
        data=file.file,
        length=file.size
    )

    await TaskService.insert_task_with_features(
        callback_url=callback_url,
        task_uuid=task_uuid,
        features_to_process=features_to_process
    )

    task_payload = TaskPayload(
        task_uuid=task_uuid,
        file_name=task_uuid,
        bucket_name=FileBucketNames.request_files_unprocessed.value,
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

@router.get(
    path="/status",
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
    task: TaskRead = await TaskService.get_task(task_uuid=task_uuid_value)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with task_uuid={task_uuid_value} doesn't exist."
        )

    task_result_data: TaskResultRead | None = await TaskService.get_task_result(task_id=task.id)

    if task.status == TaskStatus.READY and task_result_data is not None:
        status_message = f"Task has finished successfully with available result data."
    elif task.status == TaskStatus.PROCESSING and task_result_data is None:
        status_message = f"Task is still processing. No results available currently."
    elif task.status == TaskStatus.FAILED:
        status_message = f"Task has failed during internal processing."

    return TaskStatusResponse(
        status=task.status,
        status_message=status_message,
        task_uuid=task_uuid_value,
        result=task_result_data.result_data if task_result_data else None,
    )


@router.post(
    path="/delete_audio_file",
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
        await file_client.remove_object(FileBucketNames.request_files_unprocessed.value, str(task_uuid))

    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to delete task audio file task_uuid={task_uuid}. File doesn't exist in the storage."
        )
