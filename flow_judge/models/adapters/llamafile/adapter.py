"""Llamafile API adapter for external deployments."""

import asyncio
import logging
from typing import Any

from ..base import AsyncBaseAPIAdapter, BaseAPIAdapter

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class LlamafileAPIAdapter(BaseAPIAdapter):
    """API adapter for communicating with externally deployed Llamafile servers."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "not-needed",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ):
        """Initialize the Llamafile API adapter.

        :param base_url: The base URL of the deployed Llamafile server
        :param api_key: API key for authentication
        :param timeout: Request timeout in seconds
        :param max_retries: Maximum number of retry attempts for failed requests
        :param retry_delay: Delay between retry attempts in seconds
        :param kwargs: Additional keyword arguments
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required for LlamafileAPIAdapter. "
                "Install with: pip install requests"
            )

        super().__init__(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })

    def _fetch_response(self, conversation: list[dict[str, str]]) -> str:
        """Fetch a single response from the Llamafile server.

        :param conversation: List of message dictionaries with 'role' and 'content' keys
        :return: Generated response text
        :raises requests.RequestException: If the request fails after all retries
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": "flow-judge",  # Default model name for Llamafile
            "messages": conversation,
            "stream": False,
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()
                return data["choices"][0]["message"]["content"].strip()

            except requests.RequestException as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {e}"
                    )
                    asyncio.sleep(self.retry_delay)
                    continue
                else:
                    logger.error(
                        f"Request failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise

    def _fetch_batched_response(
        self, conversations: list[list[dict[str, str]]]
    ) -> list[str]:
        """Fetch responses for multiple conversations.

        :param conversations: List of conversation lists
        :return: List of generated response texts
        """
        # For now, we'll process sequentially. In the future, this could be parallelized
        # or use batch endpoints if the Llamafile server supports them
        results = []
        for conversation in conversations:
            result = self._fetch_response(conversation)
            results.append(result)
        return results

    def fetch_response(self, conversation: list[dict[str, str]]) -> str:
        """Public method to fetch a response."""
        return self._fetch_response(conversation)

    def fetch_batched_response(
        self, conversations: list[list[dict[str, str]]]
    ) -> list[str]:
        """Public method to fetch batched responses."""
        return self._fetch_batched_response(conversations)

    def health_check(self) -> bool:
        """Check if the Llamafile server is healthy and responding.

        :return: True if the server is healthy, False otherwise
        """
        try:
            url = f"{self.base_url.rstrip('/')}/models"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def __del__(self):
        """Clean up the session when the adapter is destroyed."""
        if hasattr(self, 'session'):
            self.session.close()


class AsyncLlamafileAPIAdapter(AsyncBaseAPIAdapter):
    """Async API adapter for communicating with externally deployed Llamafile servers."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "not-needed",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent_requests: int = 10,
        **kwargs: Any,
    ):
        """Initialize the async Llamafile API adapter.

        :param base_url: The base URL of the deployed Llamafile server
        :param api_key: API key for authentication
        :param timeout: Request timeout in seconds
        :param max_retries: Maximum number of retry attempts
        :param retry_delay: Delay between retry attempts
        :param max_concurrent_requests: Maximum number of concurrent requests
        :param kwargs: Additional keyword arguments
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for AsyncLlamafileAPIAdapter. "
                "Install with: pip install aiohttp"
            )

        super().__init__(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        self.session = None  # Will be created when needed

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an async HTTP session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        return self.session

    async def _async_fetch_response(self, conversation: list[dict[str, str]]) -> str:
        """Fetch a single response asynchronously.

        :param conversation: List of message dictionaries
        :return: Generated response text
        """
        async with self.semaphore:
            session = await self._get_session()
            url = f"{self.base_url.rstrip('/')}/chat/completions"

            payload = {
                "model": "flow-judge",
                "messages": conversation,
                "stream": False,
            }

            for attempt in range(self.max_retries + 1):
                try:
                    async with session.post(url, json=payload) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return data["choices"][0]["message"]["content"].strip()

                except aiohttp.ClientError as e:
                    if attempt < self.max_retries:
                        logger.warning(
                            f"Async request failed (attempt {attempt + 1}/"
                            f"{self.max_retries + 1}): {e}"
                        )
                        await asyncio.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error(
                            f"Async request failed after "
                            f"{self.max_retries + 1} attempts: {e}"
                        )
                        raise

    async def _async_fetch_batched_response(
        self, conversations: list[list[dict[str, str]]]
    ) -> list[str]:
        """Fetch responses for multiple conversations asynchronously.

        :param conversations: List of conversation lists
        :return: List of generated response texts
        """
        tasks = [self._async_fetch_response(conv) for conv in conversations]
        return await asyncio.gather(*tasks)

    async def async_fetch_response(self, conversation: list[dict[str, str]]) -> str:
        """Public method to fetch a response asynchronously."""
        return await self._async_fetch_response(conversation)

    async def async_fetch_batched_response(
        self, conversations: list[list[dict[str, str]]]
    ) -> list[str]:
        """Public method to fetch batched responses asynchronously."""
        return await self._async_fetch_batched_response(conversations)

    async def health_check(self) -> bool:
        """Check if the Llamafile server is healthy asynchronously.

        :return: True if the server is healthy, False otherwise
        """
        try:
            session = await self._get_session()
            url = f"{self.base_url.rstrip('/')}/models"
            async with session.get(url) as response:
                response.raise_for_status()
                return True
        except aiohttp.ClientError:
            return False

    async def close(self):
        """Close the async session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
