"""Core MCP server implementation using the Model Context Protocol."""

import logging
from asyncio.exceptions import CancelledError
from collections.abc import Generator
from typing import Any

import httpx
from anyio import WouldBlock
from fastmcp import FastMCP
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS

from . import __version__
from .api_client import FabricApiClient  # Re-export for test compatibility
from .config import get_default_model
from .constants import DEFAULT_MCP_HTTP_PATH, DEFAULT_MODEL, DEFAULT_VENDOR
from .fabric_prompts import FabricPatternPromptProvider
from .fabric_tools import FabricToolsMixin
from .models import PatternExecutionConfig
from .sse_parser import SSEParserMixin
from .utils import raise_mcp_error
from .validation import ValidationMixin

# Re-export for test compatibility
__all__ = ["FabricMCP", "FabricApiClient"]


class FabricMCP(FastMCP[None], FabricToolsMixin, SSEParserMixin, ValidationMixin):
    """Base class for the Model Context Protocol server."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize the MCP server with a model."""
        super().__init__(f"Fabric MCP v{__version__}")  # pyright: ignore[reportUnknownMemberType]
        self.logger = logging.getLogger(__name__)
        self.log_level = log_level

        # Load default model configuration from Fabric environment
        self._default_model: str | None = None
        self._default_vendor: str | None = None
        self._load_default_config()

        # Expose Fabric patterns as MCP prompt templates.
        self.add_provider(FabricPatternPromptProvider(self))

        # Explicitly register tool methods
        for fn in (
            self.fabric_list_patterns,
            self.fabric_get_pattern_details,
            self.fabric_run_pattern,
            self.fabric_list_models,
            self.fabric_list_strategies,
            self.fabric_get_configuration,
        ):
            self.tool(fn)

    def _load_default_config(self) -> None:
        """Load default model configuration from Fabric environment.

        This method loads DEFAULT_MODEL and DEFAULT_VENDOR from the Fabric
        environment configuration (~/.config/fabric/.env) and stores them
        as instance variables for use in pattern execution.

        Errors during configuration loading are logged but do not prevent
        server startup to ensure graceful degradation.
        """
        try:
            self._default_model, self._default_vendor = get_default_model()
            if self._default_model and self._default_vendor:
                self.logger.info(
                    "Loaded default model configuration: %s (%s)",
                    self._default_model,
                    self._default_vendor,
                )
            elif self._default_model:
                self.logger.info(
                    "Loaded ONLY default model: %s (no vendor)", self._default_model
                )
            elif self._default_vendor:
                self.logger.info(
                    "Loaded ONLY default vendor: %s (no model)", self._default_vendor
                )
            else:
                self.logger.info("No default model configuration found")
        except (OSError, ValueError, TypeError) as e:
            self.logger.warning(
                "Failed to load default model configuration: %s. "
                "Pattern execution will use hardcoded defaults.",
                e,
            )

    def get_vendor_and_model(self, config: PatternExecutionConfig) -> tuple[str, str]:
        """Get the vendor and model based on the provided configuration."""
        vendor_name = config.vendor_name or self._default_vendor
        if not vendor_name:
            logger = logging.getLogger(__name__)
            logger.debug(
                "Vendor name is None or empty. Set to hardcoded default vendor: %s",
                DEFAULT_VENDOR,
            )
            vendor_name = DEFAULT_VENDOR

        model_name = config.model_name or self._default_model
        if not model_name:
            logger = logging.getLogger(__name__)
            logger.debug(
                "Model name is None or empty. Set to hardcoded default model: %s",
                DEFAULT_MODEL,
            )
            model_name = DEFAULT_MODEL

        return vendor_name, model_name

    def _execute_fabric_pattern(
        self,
        pattern_name: str,
        input_text: str,
        config: PatternExecutionConfig | None,
        stream: bool = False,
    ) -> dict[Any, Any] | Generator[dict[str, Any], None, None]:
        """
        Execute a Fabric pattern against the API.

        Separated from the tool method to reduce complexity.
        """
        # AC5: Client-side validation
        if not pattern_name or not pattern_name.strip():
            raise ValueError("pattern_name is required and cannot be empty")

        # Use default config if none provided
        if config is None:
            config = PatternExecutionConfig()

        vendor, model_name = self.get_vendor_and_model(config)

        # AC3: Construct proper JSON payload for Fabric API /chat endpoint
        prompt_data: dict[str, Any] = {
            "userInput": input_text,
            "patternName": pattern_name.strip(),
            "model": model_name,
            "vendor": vendor,
            "contextName": "",
            "strategyName": config.strategy_name or "",
        }

        # Add variables if provided
        if config.variables is not None:
            prompt_data["variables"] = config.variables

        # Add attachments if provided
        if config.attachments is not None:
            prompt_data["attachments"] = config.attachments

        request_payload = {
            "prompts": [prompt_data],
            "language": "en",
            "temperature": config.temperature
            if config.temperature is not None
            else 0.7,
            "topP": config.top_p if config.top_p is not None else 0.9,
            "frequencyPenalty": config.frequency_penalty
            if config.frequency_penalty is not None
            else 0.0,
            "presencePenalty": config.presence_penalty
            if config.presence_penalty is not None
            else 0.0,
        }

        # AC1: Use FabricApiClient to call Fabric's /chat endpoint
        api_client = FabricApiClient()
        try:
            # AC4: Handle Server-Sent Events (SSE) stream response
            response = api_client.post("/chat", json_data=request_payload)
            response.raise_for_status()  # Raise HTTPError for bad responses

            if stream:
                # Return generator for streaming mode
                return self._parse_sse_stream(response)
            # Return accumulated result for non-streaming mode
            return self._parse_sse_response(response)

        except httpx.ConnectError as e:
            logger = logging.getLogger(__name__)
            logger.error("Failed to connect to Fabric API: %s", e)
            raise ConnectionError(f"Unable to connect to Fabric API: {e}") from e
        except httpx.HTTPStatusError as e:
            logger = logging.getLogger(__name__)
            logger.error("Fabric API HTTP error: %s", e)
            error_text = e.response.text
            status_code = e.response.status_code
            raise RuntimeError(
                f"Fabric API returned error {status_code}: {error_text}"
            ) from e
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("Unexpected error calling Fabric API: %s", e)
            raise RuntimeError(f"Unexpected error executing pattern: {e}") from e
        finally:
            api_client.close()

    def fabric_run_pattern(
        self,
        pattern_name: str,
        input_text: str = "",
        stream: bool = False,
        config: PatternExecutionConfig | None = None,
        model_name: str | None = None,
        vendor_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        strategy_name: str | None = None,
        variables: dict[str, str] | None = None,
        attachments: list[str] | None = None,
    ) -> dict[str, Any] | Generator[dict[str, Any], None, None]:
        """
        Execute a Fabric pattern with input text and return output.

        This tool calls the Fabric API's /chat endpoint to execute a named pattern
        with the provided input text. Returns either complete output (non-streaming)
        or a generator of chunks (streaming) based on the stream parameter.

        Args:
            pattern_name: The name of the fabric pattern to run (required).
            input_text: The input text to be processed by the pattern (optional).
            stream: Whether to stream the output. If True, returns a generator of
            chunks.
            config: Optional configuration for execution parameters.
            model_name: Optional model name override (e.g., "gpt-4", "claude-3-opus").
            vendor_name: Optional vendor name override (e.g., "openai", "anthropic").
            temperature: Optional temperature for LLM (0.0-2.0, controls randomness).
            top_p: Optional top-p for LLM (0.0-1.0, nucleus sampling).
            presence_penalty: Optional presence penalty (-2.0-2.0, reduces repetition).
            frequency_penalty: Optional frequency penalty (-2.0-2.0, reduces frequency).
            strategy_name: Optional strategy name for pattern execution.
            variables: Optional map of key-value strings for pattern variables.
            attachments: Optional list of file paths/URLs to attach to the pattern.

        Returns:
            dict[Any, Any] | Generator: For non-streaming, returns dict with
            'output_format' and 'output_text'.
            For streaming, returns a generator yielding dict chunks
            with 'type', 'format', and 'content'.

        Raises:
            McpError: For any API errors, connection issues, or parsing problems.
        """

        # Validate new parameters
        self._validate_execution_parameters(
            model_name,
            vendor_name,
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            strategy_name,
            variables,
            attachments,
        )

        # Merge parameters with config
        merged_config = self._merge_execution_config(
            config,
            model_name,
            vendor_name,
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            strategy_name,
            variables,
            attachments,
        )

        try:
            return self._execute_fabric_pattern(
                pattern_name, input_text, merged_config, stream
            )
        except RuntimeError as e:
            error_message = str(e)
            # Check for pattern not found (500 with file not found message)
            if (
                "Fabric API returned error 500" in error_message
                and "no such file or directory" in error_message
            ):
                raise_mcp_error(
                    e, INVALID_PARAMS, f"Pattern '{pattern_name}' not found"
                )
            # Check for other HTTP status errors
            if "Fabric API returned error" in error_message:
                raise_mcp_error(
                    e,
                    INTERNAL_ERROR,
                    f"Error executing pattern '{pattern_name}': {error_message}",
                )
            # Other runtime errors
            raise_mcp_error(
                e,
                INTERNAL_ERROR,
                f"Error executing pattern '{pattern_name}': {error_message}",
            )
        except ConnectionError as e:
            raise_mcp_error(
                e,
                INTERNAL_ERROR,
                f"Error executing pattern '{pattern_name}': {e}",
            )
        except ValueError as e:
            raise_mcp_error(
                e,
                INVALID_PARAMS,
                f"Invalid parameter for pattern '{pattern_name}': {e}",
            )

    def get_default_model_config(self) -> tuple[str | None, str | None]:
        """Get the current default model configuration.

        Returns:
            Tuple of (default_model, default_vendor). Either or both can be None.

        Note:
            This method is primarily intended for testing and introspection.
        """
        return self._default_model, self._default_vendor

    def http_streamable(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        mcp_path: str = DEFAULT_MCP_HTTP_PATH,
    ):
        """Run the MCP server with StreamableHttpTransport."""
        try:
            self.run(transport="streamable-http", host=host, port=port, path=mcp_path)
        except (KeyboardInterrupt, CancelledError, WouldBlock) as e:
            # Handle graceful shutdown
            self.logger.debug("Exception details: %s: %s", type(e).__name__, e)
            self.logger.info("Server stopped by user.")

    def stdio(self):
        """Run the MCP server."""
        try:
            self.run()
        except (KeyboardInterrupt, CancelledError, WouldBlock):
            # Handle graceful shutdown
            self.logger.info("Server stopped by user.")
