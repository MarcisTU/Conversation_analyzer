from abc import ABC, abstractmethod
from typing import Tuple

from src.models.results import Results


class BaseService(ABC):
    """Interface for AI model services."""

    @abstractmethod
    async def inference(self, file_path: str, existing_results: Results) -> Tuple[Results, str]:
        """Async call inference entry method for async workers."""

    @abstractmethod
    def _run_inference(self, file_path: str, existing_results: Results) -> Tuple[Results, str]:
        """Async call inference logic method (synchronous) for main logic execution on GPU/CPU."""
