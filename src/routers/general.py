from fastapi import APIRouter


router = APIRouter(
    prefix="/api/v1", 
    tags=["General"],
)


@router.get("/")
async def root():
    return {"message": "I am alive and well. Thank you for checking in!"}
