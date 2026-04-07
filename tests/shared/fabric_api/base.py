"Base classes for unit and integration tests with common fixtures and utilities."

import pytest
import pytest_asyncio

from fabric_mcp.core import FabricMCP


class TestFixturesBase:
    "Base class with common fixtures"

    @pytest.fixture
    def server(self):
        """Create a FabricMCP server instance for testing."""
        return FabricMCP(log_level="DEBUG")

    @pytest_asyncio.fixture
    async def mcp_tools(self, server: FabricMCP):
        """Get the MCP tools from the server."""
        tools = await server.list_tools()
        return {t.name: t for t in tools}
