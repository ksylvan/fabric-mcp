"""Unified integration tests for all MCP transport types.

This module tests HTTP Streamable, SSE, and other transport functionality
in a DRY manner, avoiding code duplication across transport types.
"""

import asyncio
import json
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from fastmcp.utilities.types import ContentBlock

from tests.shared.fabric_api.server import MOCK_PATTERNS
from tests.shared.fabric_api.utils import (
    MockFabricAPIServer,
    fabric_api_server_fixture,
)
from tests.shared.port_utils import find_free_port
from tests.shared.transport_test_utils import (
    ServerConfig,
    get_expected_tools,
    run_server,
)

_ = fabric_api_server_fixture  # eliminate unused variable warning

INVALID_PORT = 9999  # Port used for testing invalid configurations


class TransportTestBase:
    """Base class for transport-specific test configurations."""

    @pytest.fixture(scope="class")
    def server_config(self) -> ServerConfig:
        """Override in subclasses to provide transport-specific config."""
        raise NotImplementedError

    @property
    def transport_type(self) -> str:
        """Override in subclasses to specify transport type."""
        raise NotImplementedError

    def create_client(self, url: str) -> Client[Any]:
        """Override in subclasses to create transport-specific client."""
        raise NotImplementedError

    def get_server_url(self, config: ServerConfig) -> str:
        """Override in subclasses to build transport-specific URL."""
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_server_starts_and_responds(
        self, server_config: ServerConfig
    ) -> None:
        """Test that server starts and responds to basic requests."""
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)

            async with httpx.AsyncClient() as http_client:
                # HTTP endpoints return normal responses
                response = await http_client.get(url)
                if self.transport_type == "http":
                    # Expect 307 redirect or 406 without proper headers
                    assert response.status_code in [307, 406]
                    if response.status_code == 406:
                        assert (
                            "text/event-stream" in response.json()["error"]["message"]
                        )

    @pytest.mark.asyncio
    async def test_mcp_client_connection(self, server_config: ServerConfig) -> None:
        """Test MCP client can connect and list tools."""
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                tools = await client.list_tools()
                assert tools is not None
                assert isinstance(tools, list)

                # Verify expected tools are present
                tool_names: list[str] = [tool.name for tool in tools]
                expected_tools = get_expected_tools()

                for expected_tool in expected_tools:
                    assert expected_tool in tool_names

    @pytest.mark.asyncio
    async def test_fabric_list_patterns_tool_fail(
        self, server_config: ServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test fabric_list_patterns tool.

        Expects connection error when Fabric API unavailable.
        """
        # Override environment to point to non-existent Fabric API
        monkeypatch.setenv("FABRIC_BASE_URL", f"http://localhost:{INVALID_PORT}")
        monkeypatch.setenv("FABRIC_API_KEY", "test")

        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Since we don't have a real Fabric API running, we expect a ToolError
                with pytest.raises(ToolError) as exc_info:
                    await client.call_tool("fabric_list_patterns")

                # Verify it's the expected connection error
                error_msg = str(exc_info.value)
                assert (
                    "Failed to connect to Fabric API" in error_msg
                    or "Connection refused" in error_msg
                )

    @pytest.mark.asyncio
    async def test_fabric_get_pattern_details_tool(
        self, server_config: ServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test fabric_get_pattern_details tool.

        Expects connection error when Fabric API unavailable.
        """
        # Override environment to point to non-existent Fabric API
        monkeypatch.setenv("FABRIC_BASE_URL", "http://localhost:99999")
        monkeypatch.setenv("FABRIC_API_KEY", "test")

        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Since we don't have a real Fabric API running, we expect a ToolError
                with pytest.raises(ToolError) as exc_info:
                    await client.call_tool(
                        "fabric_get_pattern_details", {"pattern_name": "test_pattern"}
                    )

                # Verify it's the expected connection error
                error_msg = str(exc_info.value)
                assert "Failed to connect to Fabric API" in error_msg

    def _validate_pattern_run_result(self, result: Sequence[ContentBlock]) -> None:
        """Helper to validate pattern run result structure."""
        assert isinstance(result, list)  # type: ignore[arg-type]
        assert len(result) > 0
        assert hasattr(result[0], "text")
        assert isinstance(getattr(result[0], "text"), str)
        assert len(getattr(result[0], "text")) > 0

    @pytest.mark.asyncio
    async def test_fabric_run_pattern_tool(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_run_pattern tool (non-streaming)."""
        _ = mock_fabric_api_server  # eliminate unused variable warning
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                result = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                        "stream": False,
                    },
                )
                assert result is not None
                assert not result.is_error
                self._validate_pattern_run_result(result.content)

    @pytest.mark.asyncio
    async def test_fabric_run_pattern_streaming_tool(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_run_pattern tool with streaming."""
        _ = mock_fabric_api_server  # eliminate unused variable warning
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                result = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                        "stream": True,
                    },
                )
                assert result is not None
                assert not result.is_error

                assert len(result.content) > 0
                self._validate_pattern_run_result(result.content)

    @pytest.mark.asyncio
    async def test_fabric_run_pattern_with_model_name(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_run_pattern tool with model_name parameter (Story 3.3)."""
        _ = mock_fabric_api_server  # eliminate unused variable warning
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Test with different model names
                result_gpt4 = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                        "model_name": "gpt-4",
                    },
                )
                self._validate_pattern_run_result(result_gpt4.content)
                output_text_gpt4 = getattr(result_gpt4.content[0], "text")
                assert "gpt-4" in output_text_gpt4 or "openai" in output_text_gpt4

                result_claude = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                        "model_name": "claude-3-opus",
                    },
                )
                self._validate_pattern_run_result(result_claude.content)
                output_text_claude = getattr(result_claude.content[0], "text")
                assert "claude-3-opus" in output_text_claude

    @pytest.mark.asyncio
    async def test_fabric_run_pattern_with_strategy_name(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_run_pattern tool with strategy_name parameter (Story 3.3)."""
        _ = mock_fabric_api_server  # eliminate unused variable warning
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Test with different strategy names
                result_creative = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                        "strategy_name": "creative",
                    },
                )
                self._validate_pattern_run_result(result_creative.content)
                output_text_creative = getattr(result_creative.content[0], "text")
                assert isinstance(output_text_creative, str)
                assert (
                    "creative" in output_text_creative.lower()
                    or "Creative strategy applied" in output_text_creative
                )

                result_analytical = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                        "strategy_name": "analytical",
                    },
                )
                self._validate_pattern_run_result(result_analytical.content)
                output_text_analytical = getattr(result_analytical.content[0], "text")
                assert isinstance(output_text_analytical, str)
                output_text_analytical: str = getattr(
                    result_analytical.content[0], "text"
                )
                assert (
                    "analytical" in output_text_analytical.lower()
                    or "Analytical strategy applied" in output_text_analytical
                )

    @pytest.mark.asyncio
    async def test_fabric_run_pattern_with_llm_parameters(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_run_pattern tool with LLM tuning parameters (Story 3.3)."""
        _ = mock_fabric_api_server  # eliminate unused variable warning
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Test with temperature parameter
                result_temp = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                        "temperature": 1.5,
                    },
                )
                self._validate_pattern_run_result(result_temp.content)
                output_text_temp = getattr(result_temp.content[0], "text")
                assert "temp=1.5" in output_text_temp

                # Test with top_p parameter
                result_top_p = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                        "top_p": 0.8,
                    },
                )
                self._validate_pattern_run_result(result_top_p.content)
                output_text_top_p = getattr(result_top_p.content[0], "text")
                assert "top_p=0.8" in output_text_top_p

    @pytest.mark.asyncio
    async def test_fabric_run_pattern_parameter_combinations(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_run_pattern tool with multiple parameters (Story 3.3)."""
        _ = mock_fabric_api_server  # eliminate unused variable warning
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Test all parameters together
                result = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                        "model_name": "gpt-4",
                        "temperature": 0.85,
                        "top_p": 0.95,
                        "presence_penalty": 0.1,
                        "frequency_penalty": -0.1,
                        "strategy_name": "creative",
                    },
                )

                self._validate_pattern_run_result(result.content)
                output_text = getattr(result.content[0], "text")

                # Verify multiple parameters are reflected in output
                assert "gpt-4" in output_text
                assert "temp=0.8" in output_text
                assert "creative" in output_text.lower()

    @pytest.mark.asyncio
    async def test_fabric_run_pattern_parameter_validation_errors(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_run_pattern tool parameter validation (Story 3.3)."""
        _ = mock_fabric_api_server  # eliminate unused variable warning
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Test invalid temperature range
                with pytest.raises(ToolError) as exc_info:
                    await client.call_tool(
                        "fabric_run_pattern",
                        {
                            "pattern_name": "test_pattern",
                            "input_text": "test input",
                            "temperature": 3.0,  # Invalid: > 2.0
                        },
                    )
                assert "temperature must be a number between 0.0 and 2.0" in str(
                    exc_info.value
                )

                # Test invalid top_p range
                with pytest.raises(ToolError) as exc_info:
                    await client.call_tool(
                        "fabric_run_pattern",
                        {
                            "pattern_name": "test_pattern",
                            "input_text": "test input",
                            "top_p": 1.5,  # Invalid: > 1.0
                        },
                    )
                assert "top_p must be a number between 0.0 and 1.0" in str(
                    exc_info.value
                )

                # Test invalid model name (empty string)
                with pytest.raises(ToolError) as exc_info:
                    await client.call_tool(
                        "fabric_run_pattern",
                        {
                            "pattern_name": "test_pattern",
                            "input_text": "test input",
                            "model_name": "",  # Invalid: empty string
                        },
                    )
                assert "model_name must be a non-empty string" in str(exc_info.value)

                # Test invalid strategy name (empty string)
                with pytest.raises(ToolError) as exc_info:
                    await client.call_tool(
                        "fabric_run_pattern",
                        {
                            "pattern_name": "test_pattern",
                            "input_text": "test input",
                            "strategy_name": "",  # Invalid: empty string
                        },
                    )
                assert "strategy_name must be a non-empty string" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fabric_run_pattern_backward_compatibility(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_run_pattern tool backward compatibility (Story 3.3)."""
        _ = mock_fabric_api_server  # eliminate unused variable warning
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Test original Story 3.1 call format (no new parameters)
                result_basic = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                    },
                )
                assert result_basic is not None
                output_text_basic = getattr(result_basic.content[0], "text")
                assert len(output_text_basic) > 0

                # Test with stream parameter (original format)
                result_stream = await client.call_tool(
                    "fabric_run_pattern",
                    {
                        "pattern_name": "test_pattern",
                        "input_text": "test input",
                        "stream": False,
                    },
                )
                assert result_stream is not None
                output_text_stream = getattr(result_stream.content[0], "text")
                assert len(output_text_stream) > 0

                # Both should work identically when no new parameters are used
                # (Exact output may vary due to mock randomness)

    @pytest.mark.asyncio
    async def test_fabric_list_models_tool(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_list_models tool."""
        _ = mock_fabric_api_server  # eliminate unused variable warning

        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                result = await client.call_tool("fabric_list_models")
                assert result is not None
                assert not result.is_error

                self._validate_pattern_run_result(result.content)
                models_text = getattr(result.content[0], "text")
                assert isinstance(models_text, str)
                assert len(models_text) > 0

                # Validate the JSON structure

                models_data = json.loads(models_text)
                assert "models" in models_data
                assert "vendors" in models_data
                assert isinstance(models_data["models"], list)
                assert isinstance(models_data["vendors"], dict)

                # Verify some expected models are present
                models = models_data["models"]
                assert len(models) > 0
                assert any("gpt" in model for model in models)
                assert any("claude" in model for model in models)

                # Verify vendor structure
                vendors = models_data["vendors"]
                assert "openai" in vendors
                assert "anthropic" in vendors
                assert isinstance(vendors["openai"], list)
                assert isinstance(vendors["anthropic"], list)

    @pytest.mark.asyncio
    async def test_fabric_list_strategies_tool(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_list_strategies tool."""
        _ = mock_fabric_api_server  # eliminate unused variable warning

        # Environment is automatically configured by fixture
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                result = await client.call_tool("fabric_list_strategies")
                assert result is not None
                assert not result.is_error

                strategies_text = getattr(result.content[0], "text")
                assert isinstance(strategies_text, str)
                assert len(strategies_text) > 0

    @pytest.mark.asyncio
    async def test_fabric_get_configuration_tool(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_get_configuration tool."""
        _ = mock_fabric_api_server  # eliminate unused variable warning
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                result = await client.call_tool("fabric_get_configuration")
                assert result is not None
                assert not result.is_error

                config_text = getattr(result.content[0], "text")
                assert isinstance(config_text, str)
                # Should have redacted sensitive values
                assert "[REDACTED_BY_MCP_SERVER]" in config_text

    @pytest.mark.asyncio
    async def test_mcp_error_handling(self, server_config: ServerConfig) -> None:
        """Test MCP error handling."""
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Test calling non-existent tool
                with pytest.raises(Exception):  # Should raise MCP error
                    await client.call_tool("non_existent_tool")

    @pytest.mark.asyncio
    async def test_concurrent_requests(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test handling multiple concurrent requests."""
        _ = mock_fabric_api_server  # eliminate unused variable warning
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Make multiple concurrent requests
                tasks: list[Any] = []
                for _ in range(5):
                    task = asyncio.create_task(client.call_tool("fabric_list_patterns"))
                    tasks.append(task)

                results: list[Any] = await asyncio.gather(*tasks)

                # All requests should succeed
                for result in results:
                    assert result is not None
                    assert not result.is_error
                    assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_fabric_list_patterns_tool_success(
        self, server_config: ServerConfig, mock_fabric_api_server: MockFabricAPIServer
    ) -> None:
        """Test fabric_list_patterns tool success path.

        Uses mock Fabric API server to test successful pattern retrieval.
        """
        _ = mock_fabric_api_server  # eliminate unused variable warning

        # Environment is automatically configured by fixture
        async with run_server(server_config, self.transport_type) as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                # Call the tool and expect success
                result = await client.call_tool("fabric_list_patterns")

                # Verify response structure
                assert result is not None
                assert not result.is_error
                assert len(result.content) == 1

                # Extract the JSON text and parse it
                patterns_text = getattr(result.content[0], "text")
                assert isinstance(patterns_text, str)

                patterns: list[str] = json.loads(patterns_text)
                assert isinstance(patterns, list)
                assert len(patterns) > 0

                # Expected patterns from mock server
                expected_patterns = MOCK_PATTERNS

                # Verify all expected patterns are present
                assert patterns == expected_patterns


@pytest.mark.integration
class TestHTTPStreamableTransport(TransportTestBase):
    """Integration tests for HTTP Streamable Transport."""

    @pytest.fixture
    def server_config(self) -> ServerConfig:
        """Configuration for the HTTP server."""
        return {
            "host": "127.0.0.1",
            "port": find_free_port(),
            "mcp_path": "/message",
        }

    @property
    def transport_type(self) -> str:
        """HTTP streamable transport type."""
        return "http"

    def create_client(self, url: str) -> Client[Any]:
        """Create HTTP streamable transport client."""
        transport = StreamableHttpTransport(url=url)
        return Client(transport)

    def get_server_url(self, config: ServerConfig) -> str:
        """Build HTTP server URL."""
        return f"http://{config['host']}:{config['port']}{config['mcp_path']}"

    @pytest.mark.asyncio
    async def test_custom_host_port_path_configuration(self) -> None:
        """Test server with custom host, port, and path configuration."""
        custom_config: ServerConfig = {
            "host": "127.0.0.1",
            "port": find_free_port(),
            "mcp_path": "/custom-path",
        }

        async with run_server(custom_config, "http") as config:
            url = self.get_server_url(config)
            client = self.create_client(url)

            async with client:
                tools = await client.list_tools()
                assert tools is not None
                assert isinstance(tools, list)


@pytest.mark.integration
class TestTransportCLI:
    """Integration tests for CLI with different transports."""

    def test_cli_transport_help(self) -> None:
        """Test CLI shows transport options in help."""
        result = subprocess.run(
            [sys.executable, "-m", "fabric_mcp.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "--transport" in result.stdout
        assert "[stdio|http]" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--mcp-path" in result.stdout

    def test_cli_validates_http_options_with_stdio(self) -> None:
        """Test CLI rejects HTTP options when using stdio transport."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fabric_mcp.cli",
                "--transport",
                "stdio",
                "--host",
                "custom-host",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2
        assert "only valid with --transport http" in result.stderr
