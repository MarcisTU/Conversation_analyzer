# test_db_data.py

import argparse
import json
import os
from datetime import datetime

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel import SQLModel
import asyncio
from src.db.models import Task, Feature, FeaturesInTask, TaskResultsFinal


DB_NAME = os.getenv("DB_NAME", "kt_audio_db")
DB_USER = os.getenv("DB_USER", "marcis")
DB_PASSWORD = os.getenv("DB_PASSWORD", "upenieks123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5434")


DATABASE_URL = (
    f"postgresql+asyncpg://"
    f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


engine = create_async_engine(
    DATABASE_URL,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
)


async def load_task(task_uuid: str):
    async with AsyncSessionLocal() as session:
        task_result = await session.execute(
            select(Task).where(Task.task_uuid == task_uuid)
        )

        task = task_result.scalar_one_or_none()

        if task is None:
            print(f"Task not found: {task_uuid}")
            return

        features_result = await session.execute(
            select(FeaturesInTask)
            .where(FeaturesInTask.task_id == task.id)
            .order_by(FeaturesInTask.id)
        )

        features_in_task = features_result.scalars().all()

        final_result = await session.execute(
            select(TaskResultsFinal)
            .where(TaskResultsFinal.task_id == task.id)
        )

        task_final_result = final_result.scalar_one_or_none()

        logger.info("=" * 80)
        logger.info("TASK")
        logger.info("=" * 80)

        logger.info(f"ID:           {task.id}")
        logger.info(f"UUID:         {task.task_uuid}")
        logger.info(f"Status:       {task.status}")
        logger.info(f"Callback URL: {task.callback_url}")
        logger.info(f"Created at:   {task.created_at}")
        logger.info(f"Updated at:   {task.updated_at}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("FEATURES")
        logger.info("=" * 80)

        if not features_in_task:
            logger.info("No features found.")
        else:
            for feature_task in features_in_task:
                logger.info("")
                logger.info(f"Feature ID:   {feature_task.feature_id}")
                logger.info(f"Feature name: {feature_task.feature.name}")
                logger.info(f"Status:       {feature_task.status}")
                logger.info(f"Record ID:    {feature_task.id}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL TASK RESULT")
        logger.info("=" * 80)

        if task_final_result is None:
            logger.info("No final result found.")
        else:
            logger.info(f"Result ID:    {task_final_result.id}")
            logger.info(f"Task ID:      {task_final_result.task_id}")
            logger.info(f"Created at:   {task_final_result.created_at}")

            logger.info("")
            logger.info("Result JSON:")
            logger.info(
                json.dumps(
                    task_final_result.result_data["segments"][:10],  # print or select specific segments to reduce terminal clutter
                    indent=4,
                    ensure_ascii=False,
                    default=str,
                )
            )


async def main():
    parser = argparse.ArgumentParser(
        description="Load task information and final results for specific task inspection from PostgreSQL."
    )

    parser.add_argument(
        "--task_uuid",
        default="560df756-41a0-4ed7-8b5b-e88d8acf10ed",
        help="Task UUID to query",
    )

    args = parser.parse_args()

    await load_task(args.task_uuid)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
