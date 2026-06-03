from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def custom_validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Overriding the default 422 error response.
    Transforms FastAPI's complex error array into a cleaner format.
    """
    errors = exc.errors()

    first_error = errors[0] if errors else {}
    error_field = ".".join(str(loc) for loc in first_error.get("loc", []))
    error_msg = first_error.get("msg", "Validation error")

    custom_error_payload = {
        "status": "Request validation failed",
        "error_code": status.HTTP_422_UNPROCESSABLE_CONTENT,
        "message": f"Invalid value provided for '{error_field}'. Details: {error_msg}",
        "raw_details": errors
    }

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=custom_error_payload
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers all custom exception handlers to the FastAPI application instance.
    """
    app.add_exception_handler(RequestValidationError, custom_validation_exception_handler)

    # Add more exception handlers here:
    # app.add_exception_handler(SomeOtherCustomException, another_handler)
