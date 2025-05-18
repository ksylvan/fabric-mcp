"""Core MCP server implementation using the Model Context Protocol."""

import logging
from asyncio.exceptions import CancelledError
from typing import Any, Callable

from anyio import WouldBlock
from fastmcp import FastMCP

from . import __version__


class FabricMCP(FastMCP[None]):
    """Base class for the Model Context Protocol server."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize the MCP server with a model."""
        super().__init__(f"Fabric MCP v{__version__}", log_level=log_level)
        self.mcp = self
        self.logger = logging.getLogger(__name__)
        self.__tools: list[Callable[..., Any]] = []

        @self.tool()
        def fabric_list_patterns() -> list[str]:
            """Return a list of available fabric patterns from the Fabric backend."""
            try:
                from fabric_mcp.api_client import FabricApiClient
                client = FabricApiClient()
                resp = client.get("/patterns/names")
                patterns = resp.json()
                if not isinstance(patterns, list):
                    self.logger.error("Unexpected response type for /patterns/names: %s", type(patterns))
                    return []
                return patterns
            except Exception as e:
                self.logger.error(f"Failed to fetch pattern list from Fabric backend: {e}")
                return []

        self.__tools.append(fabric_list_patterns)

        @self.tool()
        def fabric_pattern_prompt(pattern_name: str) -> dict[Any, Any]:
            """Return the prompt of a specific fabric pattern."""
            try:
                from fabric_mcp.api_client import FabricApiClient
                client = FabricApiClient()
                resp = client.get(f"/patterns/{pattern_name}")
                data = resp.json()
                pattern_text = data.get("Pattern", "")
                return {
                    "name": data.get("Name", pattern_name),
                    "pattern": pattern_text,
                }
            except Exception as e:
                self.logger.error(f"Failed to fetch pattern prompt from Fabric backend: {e}")
                return {"name": pattern_name, "pattern": ""}

        self.__tools.append(fabric_pattern_prompt)

        @self.tool()
        def fabric_run_pattern(pattern_name: str, input_str: str) -> dict[Any, Any]:
            """
            Run a specific fabric pattern with the given arguments.

            Args:
                pattern_name (str): The name of the fabric pattern to run.
                input_str (str): The input string to be processed by the pattern.

            Returns:
                dict[Any, Any]: Contains the pattern name, input, and result.
            """
            # This is a placeholder for the actual implementation
            return {
                "name": pattern_name,
                "input": input_str,
                "result": "Pattern result here",
            }

        self.__tools.append(fabric_run_pattern)

    def stdio(self):
        """Run the MCP server."""
        try:
            self.mcp.run()
        except (KeyboardInterrupt, CancelledError, WouldBlock):
            # Handle graceful shutdown
            self.logger.info("Server stopped by user.")
