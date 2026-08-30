from typing import List, Optional, Union

from loguru import logger
from sqlalchemy import select

from src.db.db import get_db_session
from src.db.models import Task, Feature, FeaturesInTask
from src.models.enums import TaskStatus, FeatureName, FeatureStatus
from src.models.results import Results
from src.models.schemas import ProductCreate, ReviewCreate, ProductRead, TaskUpdate, \
    TaskRead, FeatureInTaskRead


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

            # Commit everything together
            await db.commit()
            await db.refresh(new_task)

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
            await db.commit()
            return True

    @staticmethod
    async def get_task(task_uuid: str) -> Union[TaskRead, None]:
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
    async def update_task(task_uuid: str, update_data: TaskUpdate):
        async with get_db_session() as db:
            result = await db.execute(
                select(Task).where(Task.task_uuid == task_uuid)
            )
            db_task = result.scalar_one_or_none()

            if not db_task:
                logger.error(f"Task with UUID {task_uuid} not found for database update operation.")
                return None

            update_dict = update_data.model_dump(exclude_unset=True)
            db_task.sqlmodel_update(update_dict)

            db.add(db_task)
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
