"""PAL file loading and parsing functionality."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import ValidationError

from ..exceptions.core import PALLoadError, PALValidationError
from ..models.schema import ComponentLibrary, EvaluationSuite, PromptAssembly


class Loader:
    """Handles loading and parsing of PAL files from local filesystem and URLs.

    The Loader provides unified file loading capabilities for all PAL file types:
    - Prompt assemblies (.pal)
    - Component libraries (.pal.lib)
    - Evaluation suites (.eval.yaml)

    Supports loading from both local files and remote URLs with automatic
    format validation using Pydantic models.

    Attributes:
        timeout: HTTP request timeout in seconds (default: 30.0)

    Example:
        >>> loader = Loader()
        >>>
        >>> # Load a prompt assembly
        >>> assembly = await loader.load_prompt_assembly_async("api_design.pal")
        >>>
        >>> # Load from URL
        >>> library = await loader.load_component_library_async(
        ...     "https://example.com/libs/personas.pal.lib"
        ... )
        >>>
        >>> # Use synchronous version
        >>> # evaluation = loader.load_evaluation_suite("examples/evaluation/classify_intent.eval.yaml")  # doctest: +SKIP
    """

    #: Default ceiling for remotely loaded documents (10 MiB)
    DEFAULT_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024

    def __init__(
        self, timeout: float = 30.0, max_download_bytes: int | None = None
    ) -> None:
        """Initialize the loader.

        Args:
            timeout: Timeout in seconds for HTTP requests when loading from URLs
            max_download_bytes: Maximum size accepted when loading from a URL.
                Defaults to DEFAULT_MAX_DOWNLOAD_BYTES; pass None to disable.
        """
        self.timeout = timeout
        self.max_download_bytes = (
            self.DEFAULT_MAX_DOWNLOAD_BYTES
            if max_download_bytes is None
            else max_download_bytes
        )
        self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Loader:
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client, if one was created.

        Call this when the loader was not used as an async context manager;
        the client is created lazily on the first URL load.
        """
        if self._http_client:
            await self._http_client.aclose()

    def load_prompt_assembly(self, path_or_url: str | Path) -> PromptAssembly:
        """Load and validate a .pal prompt assembly file.

        Synchronous wrapper for load_prompt_assembly_async.

        Args:
            path_or_url: Path to local .pal file or URL

        Returns:
            Validated PromptAssembly object

        Raises:
            PALLoadError: If file cannot be loaded
            PALValidationError: If file format is invalid
        """
        return asyncio.run(self.load_prompt_assembly_async(path_or_url))

    def load_component_library(self, path_or_url: str | Path) -> ComponentLibrary:
        """Load and validate a .pal.lib component library file.

        Synchronous wrapper for load_component_library_async.

        Args:
            path_or_url: Path to local .pal.lib file or URL

        Returns:
            Validated ComponentLibrary object

        Raises:
            PALLoadError: If file cannot be loaded
            PALValidationError: If file format is invalid
        """
        return asyncio.run(self.load_component_library_async(path_or_url))

    def load_evaluation_suite(self, path_or_url: str | Path) -> EvaluationSuite:
        """Load and validate a .eval.yaml evaluation suite file.

        Synchronous wrapper for load_evaluation_suite_async.

        Args:
            path_or_url: Path to local .eval.yaml file or URL

        Returns:
            Validated EvaluationSuite object

        Raises:
            PALLoadError: If file cannot be loaded
            PALValidationError: If file format is invalid
        """
        return asyncio.run(self.load_evaluation_suite_async(path_or_url))

    async def load_prompt_assembly_async(
        self, path_or_url: str | Path
    ) -> PromptAssembly:
        """Load a .pal prompt assembly file asynchronously.

        Args:
            path_or_url: Path to local .pal file or URL

        Returns:
            Validated PromptAssembly object

        Raises:
            PALLoadError: If file cannot be loaded
            PALValidationError: If file format is invalid

        Example:
            >>> import asyncio
            >>> async def example():
            ...     async with Loader() as loader:
            ...         assembly = await loader.load_prompt_assembly_async("prompt.pal")
            ...         print(f"Loaded: {assembly.id}")
            >>> # asyncio.run(example())  # doctest: +SKIP
        """
        content = await self._load_content(path_or_url)
        data = self._parse_yaml(content, path_or_url)

        try:
            return PromptAssembly.model_validate(data)
        except ValidationError as e:
            raise PALValidationError(
                f"Invalid prompt assembly format in {path_or_url}",
                context={"validation_errors": e.errors(), "path": str(path_or_url)},
            ) from e

    async def load_component_library_async(
        self, path_or_url: str | Path
    ) -> ComponentLibrary:
        """Load a .pal.lib component library file asynchronously.

        Args:
            path_or_url: Path to local .pal.lib file or URL

        Returns:
            Validated ComponentLibrary object

        Raises:
            PALLoadError: If file cannot be loaded
            PALValidationError: If file format is invalid
        """
        content = await self._load_content(path_or_url)
        data = self._parse_yaml(content, path_or_url)

        try:
            return ComponentLibrary.model_validate(data)
        except ValidationError as e:
            raise PALValidationError(
                f"Invalid component library format in {path_or_url}",
                context={"validation_errors": e.errors(), "path": str(path_or_url)},
            ) from e

    async def load_evaluation_suite_async(
        self, path_or_url: str | Path
    ) -> EvaluationSuite:
        """Load a .eval.yaml evaluation suite file asynchronously.

        Args:
            path_or_url: Path to local .eval.yaml file or URL

        Returns:
            Validated EvaluationSuite object

        Raises:
            PALLoadError: If file cannot be loaded
            PALValidationError: If file format is invalid
        """
        content = await self._load_content(path_or_url)
        data = self._parse_yaml(content, path_or_url)

        try:
            return EvaluationSuite.model_validate(data)
        except ValidationError as e:
            raise PALValidationError(
                f"Invalid evaluation suite format in {path_or_url}",
                context={"validation_errors": e.errors(), "path": str(path_or_url)},
            ) from e

    async def _load_content(self, path_or_url: str | Path) -> str:
        """Load content from file path or URL."""
        path_str = str(path_or_url)

        # Check if it's a URL
        parsed = urlparse(path_str)
        if parsed.scheme in ("http", "https"):
            return await self._load_from_url(path_str)
        return await self._load_from_file(Path(path_str))

    async def _load_from_file(self, path: Path) -> str:
        """Load content from local file."""
        try:
            # Use asyncio to avoid blocking on large files
            return await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError as e:
            raise PALLoadError(
                f"File not found: {path}", context={"path": str(path)}
            ) from e
        except PermissionError as e:
            raise PALLoadError(
                f"Permission denied reading file: {path}", context={"path": str(path)}
            ) from e
        except Exception as e:
            raise PALLoadError(
                f"Failed to read file {path}: {e}",
                context={"path": str(path), "error": str(e)},
            ) from e

    async def _load_from_url(self, url: str) -> str:
        """Load content from URL."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)

        try:
            response = await self._http_client.get(url)
            response.raise_for_status()
            self._check_download_size(response, url)
            return response.text
        except PALLoadError:
            raise
        except httpx.HTTPStatusError as e:
            raise PALLoadError(
                f"HTTP {e.response.status_code} error loading {url}: {e.response.text}",
                context={"url": url, "status_code": e.response.status_code},
            ) from e
        except httpx.RequestError as e:
            raise PALLoadError(
                f"Network error loading {url}: {e}",
                context={"url": url, "error": str(e)},
            ) from e
        except Exception as e:
            raise PALLoadError(
                f"Unexpected error loading {url}: {e}",
                context={"url": url, "error": str(e)},
            ) from e

    def _check_download_size(self, response: httpx.Response, url: str) -> None:
        """Reject remote documents larger than max_download_bytes."""
        limit = self.max_download_bytes
        if limit is None:
            return

        for size in (
            self._header_length(response),
            self._body_length(response),
        ):
            if size is not None and size > limit:
                raise PALLoadError(
                    f"Remote file exceeds the {limit} byte limit: {url} ({size} bytes)",
                    context={"url": url, "size": size, "limit": limit},
                )

    @staticmethod
    def _header_length(response: httpx.Response) -> int | None:
        """Read the declared Content-Length, if it is usable."""
        headers = getattr(response, "headers", None)
        if not isinstance(headers, httpx.Headers):
            return None

        declared = headers.get("content-length")
        try:
            return int(declared) if declared is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _body_length(response: httpx.Response) -> int | None:
        """Measure the received body, if it is usable."""
        content = getattr(response, "content", None)
        return len(content) if isinstance(content, bytes | bytearray) else None

    def _parse_yaml(self, content: str, source: str | Path) -> dict[str, Any]:
        """Parse YAML content with error handling."""
        try:
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                raise PALValidationError(
                    f"YAML content must be a dictionary, got {type(data).__name__}",
                    context={"source": str(source)},
                )
            return data
        except yaml.YAMLError as e:
            raise PALValidationError(
                f"Invalid YAML syntax in {source}: {e}",
                context={"source": str(source), "yaml_error": str(e)},
            ) from e
