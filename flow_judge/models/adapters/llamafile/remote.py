"""Remote Llamafile model for external deployments."""

import logging
from typing import Any

from flow_judge.models.common import (
    AsyncBaseFlowJudgeModel,
    BaseFlowJudgeModel,
    FlowJudgeRemoteModel,
    GenerationParams,
    ModelConfig,
    ModelType,
)

from .adapter import AsyncLlamafileAPIAdapter, LlamafileAPIAdapter

logger = logging.getLogger(__name__)


class RemoteLlamafileConfig(ModelConfig):
    """Configuration class for remote Llamafile models."""

    _DEFAULT_MODEL_ID = "flowaicom/Flow-Judge-v0.1-Llamafile-Remote"

    def __init__(
        self,
        base_url: str,
        generation_params: GenerationParams,
        api_key: str = "not-needed",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent_requests: int = 10,
        **kwargs: Any,
    ):
        """Initialize RemoteLlamafileConfig.

        :param base_url: The base URL of the deployed Llamafile server
        :param generation_params: Parameters for text generation
        :param api_key: API key for authentication
        :param timeout: Request timeout in seconds
        :param max_retries: Maximum number of retry attempts
        :param retry_delay: Delay between retry attempts
        :param max_concurrent_requests: Maximum number of concurrent requests
        :param kwargs: Additional keyword arguments
        """
        model_id = kwargs.pop("_model_id", None) or self._DEFAULT_MODEL_ID
        super().__init__(
            model_id=model_id,
            model_type=ModelType.LLAMAFILE,
            generation_params=generation_params,
            **kwargs,
        )
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent_requests = max_concurrent_requests


class RemoteLlamafile(FlowJudgeRemoteModel):
    """FlowJudge model class for remote Llamafile deployments."""

    _DEFAULT_MODEL_ID = "flowaicom/Flow-Judge-v0.1-Llamafile-Remote"
    _MODEL_TYPE = "llamafile_remote"

    def __init__(
        self,
        base_url: str,
        generation_params: dict[str, Any] | None = None,
        api_key: str = "not-needed",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ):
        """Initialize the remote Llamafile model.

        :param base_url: The base URL of the deployed Llamafile server
        :param generation_params: Dictionary of parameters for text generation
        :param api_key: API key for authentication
        :param timeout: Request timeout in seconds
        :param max_retries: Maximum number of retry attempts
        :param retry_delay: Delay between retry attempts
        :param kwargs: Additional keyword arguments
        """
        generation_params = GenerationParams(**(generation_params or {}))
        
        # Allow internal override of model_id for debugging/development
        model_id = kwargs.pop("_model_id", None) or self._DEFAULT_MODEL_ID

        # Create the API adapter
        api_adapter = LlamafileAPIAdapter(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Initialize the parent remote model
        super().__init__(
            model_id=model_id,
            model_type=self._MODEL_TYPE,
            generation_params=generation_params,
            api_adapter=api_adapter,
            **kwargs,
        )

        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def health_check(self) -> bool:
        """Check if the remote Llamafile server is healthy.

        :return: True if the server is healthy, False otherwise
        """
        return self.api_adapter.health_check()

    def _generate(self, prompt: str) -> str:
        """Generate a response using the remote Llamafile server.

        :param prompt: Input prompt
        :return: Generated response
        """
        return self.generate(prompt)

    def _batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        """Generate responses for multiple prompts.

        :param prompts: List of input prompts
        :param use_tqdm: Whether to show progress bar
        :param kwargs: Additional keyword arguments
        :return: List of generated responses
        """
        return self.batch_generate(prompts, use_tqdm=use_tqdm, **kwargs)


class AsyncRemoteLlamafile(AsyncBaseFlowJudgeModel):
    """Async FlowJudge model class for remote Llamafile deployments."""

    _DEFAULT_MODEL_ID = "flowaicom/Flow-Judge-v0.1-Llamafile-Remote-Async"
    _MODEL_TYPE = "llamafile_remote_async"

    def __init__(
        self,
        base_url: str,
        generation_params: dict[str, Any] | None = None,
        api_key: str = "not-needed",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent_requests: int = 10,
        **kwargs: Any,
    ):
        """Initialize the async remote Llamafile model.

        :param base_url: The base URL of the deployed Llamafile server
        :param generation_params: Dictionary of parameters for text generation
        :param api_key: API key for authentication
        :param timeout: Request timeout in seconds
        :param max_retries: Maximum number of retry attempts
        :param retry_delay: Delay between retry attempts
        :param max_concurrent_requests: Maximum number of concurrent requests
        :param kwargs: Additional keyword arguments
        """
        generation_params = GenerationParams(**(generation_params or {}))
        
        # Allow internal override of model_id for debugging/development
        model_id = kwargs.pop("_model_id", None) or self._DEFAULT_MODEL_ID

        # Initialize the parent async model
        super().__init__(
            model_id=model_id,
            model_type=self._MODEL_TYPE,
            generation_params=generation_params,
            **kwargs,
        )

        # Create the async API adapter
        self.api_adapter = AsyncLlamafileAPIAdapter(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_concurrent_requests=max_concurrent_requests,
        )

        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent_requests = max_concurrent_requests

    async def health_check(self) -> bool:
        """Check if the remote Llamafile server is healthy asynchronously.

        :return: True if the server is healthy, False otherwise
        """
        return await self.api_adapter.health_check()

    async def _async_generate(self, prompt: str) -> str:
        """Generate a response asynchronously.

        :param prompt: Input prompt
        :return: Generated response
        """
        conversation = [{"role": "user", "content": prompt.strip()}]
        return await self.api_adapter.async_fetch_response(conversation)

    async def _async_batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        """Generate responses for multiple prompts asynchronously.

        :param prompts: List of input prompts
        :param use_tqdm: Whether to show progress bar
        :param kwargs: Additional keyword arguments
        :return: List of generated responses
        """
        conversations = [
            [{"role": "user", "content": prompt.strip()}] for prompt in prompts
        ]
        return await self.api_adapter.async_fetch_batched_response(conversations)

    async def close(self):
        """Close the async adapter."""
        await self.api_adapter.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class CombinedRemoteLlamafile(RemoteLlamafile, AsyncRemoteLlamafile):
    """Combined remote Llamafile model supporting both sync and async operations."""

    _DEFAULT_MODEL_ID = "flowaicom/Flow-Judge-v0.1-Llamafile-Remote-Combined"
    _MODEL_TYPE = "llamafile_remote_combined"

    def __init__(
        self,
        base_url: str,
        generation_params: dict[str, Any] | None = None,
        api_key: str = "not-needed",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent_requests: int = 10,
        **kwargs: Any,
    ):
        """Initialize the combined remote Llamafile model.

        :param base_url: The base URL of the deployed Llamafile server
        :param generation_params: Dictionary of parameters for text generation
        :param api_key: API key for authentication
        :param timeout: Request timeout in seconds
        :param max_retries: Maximum number of retry attempts
        :param retry_delay: Delay between retry attempts
        :param max_concurrent_requests: Maximum number of concurrent requests
        :param kwargs: Additional keyword arguments
        """
        generation_params = GenerationParams(**(generation_params or {}))
        
        # Allow internal override of model_id for debugging/development
        model_id = kwargs.pop("_model_id", None) or self._DEFAULT_MODEL_ID

        # Initialize both parent classes
        BaseFlowJudgeModel.__init__(
            self, model_id, self._MODEL_TYPE, generation_params, **kwargs
        )
        AsyncBaseFlowJudgeModel.__init__(
            self, model_id, self._MODEL_TYPE, generation_params, **kwargs
        )

        # Create both sync and async adapters
        self.sync_adapter = LlamafileAPIAdapter(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        self.async_adapter = AsyncLlamafileAPIAdapter(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_concurrent_requests=max_concurrent_requests,
        )

        # Set the API adapter for the parent FlowJudgeRemoteModel compatibility
        self.api_adapter = self.sync_adapter

        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent_requests = max_concurrent_requests

    def health_check(self) -> bool:
        """Check if the remote Llamafile server is healthy (sync).

        :return: True if the server is healthy, False otherwise
        """
        return self.sync_adapter.health_check()

    async def async_health_check(self) -> bool:
        """Check if the remote Llamafile server is healthy (async).

        :return: True if the server is healthy, False otherwise
        """
        return await self.async_adapter.health_check()

    def _generate(self, prompt: str) -> str:
        """Generate a response using the sync adapter.

        :param prompt: Input prompt
        :return: Generated response
        """
        conversation = [{"role": "user", "content": prompt.strip()}]
        return self.sync_adapter.fetch_response(conversation)

    async def _async_generate(self, prompt: str) -> str:
        """Generate a response using the async adapter.

        :param prompt: Input prompt
        :return: Generated response
        """
        conversation = [{"role": "user", "content": prompt.strip()}]
        return await self.async_adapter.async_fetch_response(conversation)

    def _batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        """Generate responses for multiple prompts using the sync adapter.

        :param prompts: List of input prompts
        :param use_tqdm: Whether to show progress bar
        :param kwargs: Additional keyword arguments
        :return: List of generated responses
        """
        conversations = [
            [{"role": "user", "content": prompt.strip()}] for prompt in prompts
        ]
        return self.sync_adapter.fetch_batched_response(conversations)

    async def _async_batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        """Generate responses for multiple prompts using the async adapter.

        :param prompts: List of input prompts
        :param use_tqdm: Whether to show progress bar
        :param kwargs: Additional keyword arguments
        :return: List of generated responses
        """
        conversations = [
            [{"role": "user", "content": prompt.strip()}] for prompt in prompts
        ]
        return await self.async_adapter.async_fetch_batched_response(conversations)

    async def close(self):
        """Close the async adapter."""
        await self.async_adapter.close()

    def cleanup(self):
        """Clean up both adapters."""
        if hasattr(self.sync_adapter, '__del__'):
            self.sync_adapter.__del__()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup in __del__: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def __enter__(self):
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        self.cleanup()
