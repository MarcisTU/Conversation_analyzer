from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, Column, Relationship
import sqlalchemy.dialects.postgresql as pg

from src.models.enums import FeatureStatus, FeatureName
from src.models.results import Results


class Feature(SQLModel, table=True):
    __tablename__ = "features"

    id: int | None = Field(default=None, primary_key=True)
    name: FeatureName = Field(unique=True, index=True)
    order_idx: int = Field(index=True)

    def __repr__(self):
        return f"<Feature {self.id}>"


class FeaturesInTask(SQLModel, table=True):
    __tablename__ = "features_in_task"

    id: int | None = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="tasks.id", ondelete="CASCADE")
    feature_id: int = Field(foreign_key="features.id", ondelete="RESTRICT")
    status: FeatureStatus = Field(default=FeatureStatus.WAITING)
    result_data: Results | None = Field(default=None, sa_column=Column(pg.JSONB))

    # Add relationship to not have to do 2 separate queries to get feature name
    feature: Feature = Relationship(sa_relationship_kwargs={"lazy": "joined"})

    # Allow Pydantic to accept complex arbitrary types
    model_config = {
        "arbitrary_types_allowed": True
    }

    def __repr__(self):
        return f"<FeaturesInTask {self.id}>"


class Task(SQLModel, table=True):
    __tablename__ = "tasks"

    id: int | None = Field(default=None, primary_key=True)
    task_uuid: str = Field(sa_column_kwargs={"unique": True, "index": True})
    status: str
    callback_url: Optional[str] = Field(default=None)
    created_at: Optional[datetime] = Field(default=None,
                                           sa_column=Column(pg.TIMESTAMP, nullable=False, default=datetime.now))
    updated_at: Optional[datetime] = Field(default=None,
                                           sa_column=Column(pg.TIMESTAMP, nullable=False, default=datetime.now,
                                                            onupdate=datetime.now))

    def __repr__(self):
        return f"<Task {self.id}>"


class TaskResultsFinal(SQLModel, table=True):
    __tablename__ = "task_results_final"

    id: int | None = Field(default=None, primary_key=True)
    task_id: int = Field(
        foreign_key="tasks.id",
        unique=True,  # allow only one task_id to have one result (one-to-one rel)
        index=True,
        ondelete="CASCADE",
    )
    result_data: Results = Field(
        sa_column=Column(pg.JSONB, nullable=False)
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(pg.TIMESTAMP, nullable=False),
    )
