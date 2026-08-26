import os
from loguru import logger
from fastapi import HTTPException, Depends, status
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    # TODO could add api key storage in DB for multi-client setup!
    expected_api_key = os.getenv("API_KEY")

    if not expected_api_key:
        logger.error("API_KEY environment variable is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API authentication is misconfigured on the server."
        )

    if api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key."
        )

    return api_key
