import os
from contextlib import asynccontextmanager
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool

from src.db.models import Product, Review


# Create async engine
async_engine = create_async_engine(
    os.environ['DATABASE_URL'],
    poolclass=AsyncAdaptedQueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
)

# Async session factory
async_session = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# TODO we use Alembic to manage db (api service runs alembic upgrade)
# # Create all db model tables
# async def init_db():
#     async with async_engine.begin() as conn:
#         await conn.run_sync(SQLModel.metadata.create_all)


# For FastAPI dependency
async def get_db():
    async with async_session() as session:
        yield session


# For workers db usage
@asynccontextmanager
async def get_db_session():
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except:
            await session.rollback()
            raise
