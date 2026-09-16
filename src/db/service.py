from typing import List, Dict

from loguru import logger
from sqlalchemy import select
from datetime import datetime, timezone
from collections import defaultdict

from src.db.db import get_db_session
from src.db.models import Task, Feature, FeaturesInTask, TaskResultsFinal
from src.models.enums import TaskStatus, FeatureName, FeatureStatus
from src.models.results import Results
from src.models.schemas import TaskUpdate, \
    TaskRead, FeatureInTaskRead, TaskResultRead


class TaskService:
    @staticmethod
    async def insert_task_with_features(
        callback_url: str,
        task_uuid: str,
        features_to_process: list[FeatureName]
    ):
        async with get_db_session() as db:
            # Fetch the features matching the requested names to get their IDs
            feature_query = select(Feature).where(Feature.name.in_(features_to_process))
            result = await db.execute(feature_query)
            features = result.scalars().all()

            if not features:
                raise ValueError("None of the provided features were found in the database.")

            # Create and add the new Task
            new_task = Task(
                task_uuid=task_uuid,
                status=TaskStatus.WAITING.value,
                callback_url=callback_url
            )
            db.add(new_task)

            # Flush here to populate new_task.id without committing the transaction yet
            await db.flush()

            # Create FeaturesInTask link records for each found feature
            features_in_task_entries = [
                FeaturesInTask(
                    task_id=new_task.id,
                    feature_id=f.id,
                    status=FeatureStatus.WAITING,
                    result_data=None
                )
                for f in features
            ]
            db.add_all(features_in_task_entries)

    @staticmethod
    async def update_feature_status_by_type(
        task_uuid: str,
        feature_name: FeatureName,
        new_status: FeatureStatus,
        result_data: Results | None = None
    ) -> bool:
        async with get_db_session() as db:
            # Query by joining FeaturesInTask to Tasks and Features tables
            stmt = (
                select(FeaturesInTask)
                .join(Task, FeaturesInTask.task_id == Task.id)
                .join(Feature, FeaturesInTask.feature_id == Feature.id)
                .where(Task.task_uuid == task_uuid)
                .where(Feature.name == feature_name)
            )

            result = await db.execute(stmt)
            feature_in_task = result.scalars().first()

            if not feature_in_task:
                return False

            feature_in_task.status = new_status

            if result_data is not None:
                # Convert the Pydantic model to a raw dict for JSONB compatibility
                feature_in_task.result_data = result_data.model_dump(mode="json")

            db.add(feature_in_task)

            return True

    @staticmethod
    async def get_task(task_uuid: str) -> TaskRead | None:
        async with get_db_session() as db:
            result = await db.execute(
                select(Task).where(Task.task_uuid == task_uuid)
            )
            db_task = result.scalar_one_or_none()

            task_data = None
            if not db_task:
                logger.error(f"Task with UUID {task_uuid} not found in database.")
            else:
                task_data = TaskRead.model_validate(db_task)

            return task_data

    @staticmethod
    async def get_task_result(task_id: int) -> TaskResultRead | None:
        async with get_db_session() as db:
            result = await db.execute(
                select(TaskResultsFinal).where(TaskResultsFinal.task_id == task_id)
            )
            db_task_result = result.scalar_one_or_none()

            return (
                TaskResultRead.model_validate(db_task_result)
                if db_task_result
                else None
            )

    @staticmethod
    async def update_task(task_uuid: str, update_data: TaskUpdate):
        async with get_db_session() as db:
            result = await db.execute(
                select(Task).where(Task.task_uuid == task_uuid)
            )
            db_task = result.scalar_one_or_none()

            if not db_task:
                logger.error(f"Task with UUID {task_uuid} not found for database update operation.")
                return None

            update_dict = update_data.model_dump(exclude_unset=True, exclude={"results"})
            db_task.sqlmodel_update(update_dict)

            # Update/create final result
            if update_data.results is not None:
                final_result_query = await db.execute(
                    select(TaskResultsFinal)
                    .where(TaskResultsFinal.task_id == db_task.id)
                )

                final_result = final_result_query.scalar_one_or_none()

                if final_result:
                    # Final result already exists -> update it
                    final_result.result_data = (
                        update_data.results.model_dump(mode="json")
                    )
                    db.add(final_result)
                else:
                    # First final result -> create it
                    final_result = TaskResultsFinal(
                        task_id=db_task.id,
                        result_data=update_data.results.model_dump(mode="json"),
                    )
                    db.add(final_result)

            await db.flush()

            logger.info(f"Database row updated for Task UUID: {task_uuid} | Status changed to: {db_task.status}")

    @staticmethod
    async def get_features_for_task(task_uuid: str) -> list[FeatureInTaskRead]:
        """
        Retrieves all feature execution records assigned to a specific task,
        sorted by their intended execution order index. Returns a clean Pydantic schema.
        """
        async with get_db_session() as db:
            # Explicitly select both models
            stmt = (
                select(FeaturesInTask, Feature)
                .join(Task, FeaturesInTask.task_id == Task.id)
                .join(Feature, FeaturesInTask.feature_id == Feature.id)
                .where(Task.task_uuid == task_uuid)
                # Sort by the feature order_idx configuration
                .order_by(Feature.order_idx)
            )

            result = await db.execute(stmt)
            rows = result.all()  # Returns a list of tuples: [(FeaturesInTask, Feature), ...]

            return [
                FeatureInTaskRead(
                    id=f_task.id,
                    task_id=f_task.task_id,
                    feature_id=f_task.feature_id,
                    status=f_task.status,
                    result_data=f_task.result_data,
                    name=feat.name,
                    order_idx=feat.order_idx
                )
                for f_task, feat in rows
            ]

    @staticmethod
    async def get_pending_feature_requests() -> List[Dict]:
        """
        Recover pending feature requests from the database.

        A feature is considered pending when its status is WAITING.

        Only the first WAITING feature for each task is returned, based on
        Feature.order_idx, because features within a task are processed
        sequentially.
        """
        async with get_db_session() as db:
            stmt = (
                select(FeaturesInTask, Feature, Task)
                .join(Task, FeaturesInTask.task_id == Task.id)
                .join(Feature, FeaturesInTask.feature_id == Feature.id)
                .where(
                    FeaturesInTask.status.in_(
                        [FeatureStatus.WAITING, FeatureStatus.READY]
                    )
                )
                .order_by(
                    Task.id,
                    Feature.order_idx,
                )
            )

            result = await db.execute(stmt)
            rows = result.all()

            # Group features by task.
            task_features: dict[str, list[tuple]] = defaultdict(list)

            for feature_in_task, feature, task in rows:
                task_features[task.task_uuid].append(
                    (feature_in_task, feature, task)
                )

            pending_requests = []

            queued_at = datetime.now(timezone.utc).isoformat()

            for task_uuid, features in task_features.items():
                waiting_feature = None
                existing_results = None

                for feature_in_task, feature, task in features:
                    if feature_in_task.status == FeatureStatus.READY:
                        if feature_in_task.result_data is not None:
                            existing_results = feature_in_task.result_data

                    elif (
                        feature_in_task.status == FeatureStatus.WAITING
                        and waiting_feature is None
                    ):
                        waiting_feature = (
                            feature_in_task,
                            feature,
                            task,
                        )

                if waiting_feature is None:
                    continue

                feature_in_task, feature, task = waiting_feature

                pending_requests.append(
                    {
                        "request_id": task.task_uuid,
                        "worker_type": feature.name.value,
                        "callback_url": task.callback_url,
                        "queued_at": queued_at,
                        "features_in_task_id": feature_in_task.id,
                        "existing_results": existing_results,
                    }
                )

            logger.info(f"Recovered {len(pending_requests)} pending feature requests from database.")

            return pending_requests
