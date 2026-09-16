from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends

from src.modules.exception_handlers import register_exception_handlers
from src.modules.security import verify_api_key
from src.routers.tasks import router as tasks_router
from src.routers.general import router as general_router
from src.modules.managers import minio_manager, rmq_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    await rmq_manager.connect()
    minio_manager.init_client()
    yield
    await rmq_manager.close()
    await minio_manager.close_client()


# Protect all routes. If need some to be public can create separately Dependency for each
app = FastAPI(
    lifespan=lifespan, 
    dependencies=[Depends(verify_api_key)], # global dependency
)
app.include_router(tasks_router)
app.include_router(general_router)

register_exception_handlers(app)
