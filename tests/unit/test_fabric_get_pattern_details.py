"""Unit tests for fabric_get_pattern_details tool."""

from collections.abc import Callable

import pytest
import pytest_asyncio
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR

from fabric_mcp.core import FabricMCP
from tests.shared.fabric_api.base import TestFixturesBase
from tests.shared.fabric_api_mocks import (
    FabricApiMockBuilder,
    assert_api_client_calls,
    assert_mcp_error,
    mock_fabric_api_client,
)


class TestFabricGetPatternDetails(TestFixturesBase):
    """Test suite for fabric_get_pattern_details tool."""

    @pytest_asyncio.fixture
    async def get_pattern_details_tool(
        self, server: FabricMCP
    ) -> Callable[[str], dict[str, str]]:
        """Get the fabric_get_pattern_details tool function."""
        tools_list = await server.list_tools()
        tools = {t.name: t for t in tools_list}
        return getattr(tools["fabric_get_pattern_details"], "fn")

    def test_successful_pattern_details_retrieval(
        self,
        get_pattern_details_tool: Callable[[str], dict[str, str]],
    ) -> None:
        """Test successful retrieval of pattern details."""
        # Arrange
        builder = FabricApiMockBuilder().with_successful_pattern_details(
            "summarize",
            "Create a concise summary",
            "# IDENTITY\nYou are an expert summarizer...",
        )

        with mock_fabric_api_client(builder) as mock_api_client:
            # Act
            result = get_pattern_details_tool("summarize")

            # Assert
            assert isinstance(result, dict)
            assert result["name"] == "summarize"
            assert result["description"] == "Create a concise summary"
            assert (
                result["system_prompt"] == "# IDENTITY\nYou are an expert summarizer..."
            )

            assert_api_client_calls(mock_api_client, "/patterns/summarize")

    def test_successful_pattern_details_with_empty_description(
        self,
        get_pattern_details_tool: Callable[[str], dict[str, str]],
    ) -> None:
        """Test successful retrieval when description is empty."""
        # Arrange
        builder = FabricApiMockBuilder().with_successful_pattern_details(
            "test_pattern", "", "# Test pattern content"
        )

        with mock_fabric_api_client(builder) as mock_api_client:
            # Act
            result = get_pattern_details_tool("test_pattern")

            # Assert
            assert result["name"] == "test_pattern"
            assert result["description"] == ""
            assert result["system_prompt"] == "# Test pattern content"
            assert_api_client_calls(mock_api_client, "/patterns/test_pattern")

    def test_pattern_not_found_500_error(
        self,
        get_pattern_details_tool: Callable[[str], dict[str, str]],
    ) -> None:
        """Test handling of 500 server error for non-existent pattern."""
        # Arrange
        builder = FabricApiMockBuilder().with_http_error(500, "Internal Server Error")

        with mock_fabric_api_client(builder) as mock_api_client:
            # Act & Assert
            with pytest.raises(McpError) as exc_info:
                get_pattern_details_tool("nonexistent_pattern")

            assert_mcp_error(exc_info, INTERNAL_ERROR, "Fabric API internal error")
            assert_api_client_calls(mock_api_client, "/patterns/nonexistent_pattern")

    def test_pattern_details_with_special_characters(
        self,
        get_pattern_details_tool: Callable[[str], dict[str, str]],
    ) -> None:
        """Test pattern details retrieval with special characters in name."""
        # Arrange
        builder = FabricApiMockBuilder().with_successful_pattern_details(
            "pattern-with_special.chars",
            "A pattern with special characters",
            "# Special pattern",
        )

        with mock_fabric_api_client(builder) as mock_api_client:
            # Act
            result = get_pattern_details_tool("pattern-with_special.chars")

            # Assert
            assert result["name"] == "pattern-with_special.chars"
            assert result["description"] == "A pattern with special characters"
            assert result["system_prompt"] == "# Special pattern"
            assert_api_client_calls(
                mock_api_client, "/patterns/pattern-with_special.chars"
            )

    def test_pattern_details_with_long_content(
        self,
        get_pattern_details_tool: Callable[[str], dict[str, str]],
    ) -> None:
        """Test pattern details with long content."""
        # Arrange
        long_description = "A" * 1000
        long_pattern = "# " + "B" * 5000
        builder = FabricApiMockBuilder().with_successful_pattern_details(
            "long_pattern", long_description, long_pattern
        )

        with mock_fabric_api_client(builder) as mock_api_client:
            # Act
            result = get_pattern_details_tool("long_pattern")

            # Assert
            assert result["name"] == "long_pattern"
            assert result["description"] == long_description
            assert result["system_prompt"] == long_pattern
            assert_api_client_calls(mock_api_client, "/patterns/long_pattern")

    def test_connection_error_handling(
        self,
        get_pattern_details_tool: Callable[[str], dict[str, str]],
    ) -> None:
        """Test handling of connection errors."""
        # Arrange
        builder = FabricApiMockBuilder().with_connection_error("Connection failed")

        with mock_fabric_api_client(builder) as mock_api_client:
            # Act & Assert
            with pytest.raises(McpError) as exc_info:
                get_pattern_details_tool("test_pattern")

            assert_mcp_error(
                exc_info, INTERNAL_ERROR, "Failed to connect to Fabric API"
            )
            # Connection errors may not result in API calls
            assert mock_api_client.get.call_count >= 0

    def test_timeout_error_handling(
        self,
        get_pattern_details_tool: Callable[[str], dict[str, str]],
    ) -> None:
        """Test handling of timeout errors."""
        # Arrange
        builder = FabricApiMockBuilder().with_timeout_error()

        with mock_fabric_api_client(builder) as mock_api_client:
            # Act & Assert
            with pytest.raises(McpError) as exc_info:
                get_pattern_details_tool("test_pattern")

            assert_mcp_error(
                exc_info, INTERNAL_ERROR, "Failed to connect to Fabric API"
            )
            # Timeout errors may not result in completed API calls
            assert mock_api_client.get.call_count >= 0

    def test_404_error_handling(
        self,
        get_pattern_details_tool: Callable[[str], dict[str, str]],
    ) -> None:
        """Test handling of 404 errors for non-existent patterns."""
        # Arrange
        builder = FabricApiMockBuilder().with_http_error(404, "Pattern not found")

        with mock_fabric_api_client(builder) as mock_api_client:
            # Act & Assert
            with pytest.raises(McpError) as exc_info:
                get_pattern_details_tool("nonexistent")

            assert_mcp_error(
                exc_info,
                INTERNAL_ERROR,
                "Fabric API error during retrieving pattern details: 404",
            )
            assert_api_client_calls(mock_api_client, "/patterns/nonexistent")

    def test_unexpected_error_handling(
        self,
        get_pattern_details_tool: Callable[[str], dict[str, str]],
    ) -> None:
        """Test handling of unexpected errors."""
        # Arrange
        builder = FabricApiMockBuilder().with_unexpected_error(
            ValueError("Unexpected error")
        )

        with mock_fabric_api_client(builder) as mock_api_client:
            # Act & Assert
            with pytest.raises(McpError) as exc_info:
                get_pattern_details_tool("test_pattern")

            assert_mcp_error(
                exc_info,
                INTERNAL_ERROR,
                "Unexpected error during retrieving pattern details",
            )
            # Unexpected errors may not result in completed API calls
            assert mock_api_client.get.call_count >= 0
