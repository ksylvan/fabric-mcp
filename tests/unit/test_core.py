"""Test core functionality of fabric-mcp"""

import logging
import subprocess
import sys
from asyncio.exceptions import CancelledError
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from anyio import WouldBlock
from fastmcp import FastMCP
from fastmcp.tools import Tool

from fabric_mcp import __version__
from fabric_mcp.core import FabricMCP
from tests.shared.fabric_api.base import TestFixturesBase
from tests.shared.fabric_api_mocks import (
    FabricApiMockBuilder,
    mock_fabric_api_client,
)
from tests.shared.mocking_utils import COMMON_PATTERN_LIST


class TestCore(TestFixturesBase):
    """Test core functionality of fabric-mcp."""

    def test_cli_version(self):
        """Test the --version flag of the CLI."""
        command = [sys.executable, "-m", "fabric_mcp.cli", "--version"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        # click --version action prints to stdout and exits with 0
        assert result.returncode == 0
        assert result.stderr == ""
        expected_output = f"fabric-mcp, version {__version__}\n"
        assert result.stdout == expected_output

    def test_server_initialization(self, server: FabricMCP):
        """Test the initialization of the FabricMCPServer."""
        assert isinstance(server, FastMCP)
        assert server.name == f"Fabric MCP v{__version__}"
        assert isinstance(server.logger, logging.Logger)
        # Check if log level propagates (Note: FastMCP handles its own logger setup)
        # We check the logger passed during init, FastMCP might configure differently
        # assert server.logger.level == logging.DEBUG

    def test_stdio_method_runs_mcp(self, server: FabricMCP):
        """Test that the stdio method calls mcp.run()."""
        with patch.object(server, "run") as mock_run:
            server.stdio()
            mock_run.assert_called_once()

    def test_stdio_method_handles_keyboard_interrupt(
        self, server: FabricMCP, caplog: pytest.LogCaptureFixture
    ):
        """Test that stdio handles KeyboardInterrupt gracefully."""
        with patch.object(server, "run", side_effect=KeyboardInterrupt):
            with caplog.at_level(logging.INFO):
                server.stdio()
        assert "Server stopped by user." in caplog.text

    def test_stdio_method_handles_cancelled_error(
        self, server: FabricMCP, caplog: pytest.LogCaptureFixture
    ):
        """Test that stdio handles CancelledError gracefully."""
        with patch.object(server, "run", side_effect=CancelledError):
            with caplog.at_level(logging.INFO):
                server.stdio()
        assert "Server stopped by user." in caplog.text

    def test_stdio_method_handles_would_block(
        self, server: FabricMCP, caplog: pytest.LogCaptureFixture
    ):
        """Test that stdio handles WouldBlock gracefully."""
        with patch.object(server, "run", side_effect=WouldBlock):
            with caplog.at_level(logging.INFO):
                server.stdio()
        assert "Server stopped by user." in caplog.text

    def test_server_initialization_with_default_log_level(self, server: FabricMCP):
        """Test server initialization with default log level."""
        server = FabricMCP()
        assert server.log_level == "INFO"

    def test_server_initialization_with_custom_log_level(self, server: FabricMCP):
        """Test server initialization with custom log level."""
        server = FabricMCP(log_level="ERROR")
        assert server.log_level == "ERROR"

    @pytest.mark.asyncio
    async def test_fabric_mcp_tools_registration(self):
        """Test that tools are properly registered with the FastMCP instance."""
        server = FabricMCP()

        # Check that tools are available through the mcp instance
        # Note: The exact way to check registered tools may depend on FastMCP's API
        # This is a basic check to ensure the tools list is populated
        assert hasattr(server, "list_tools")
        tools = await server.list_tools()
        assert len(tools) == 6

    def test_tool_registration_coverage(self, mcp_tools: dict[str, Tool]):
        """Test that all tools are properly registered and accessible."""

        # Check that the tools are registered by accessing them
        assert len(mcp_tools) == 6

        self._test_list_patterns_tool(getattr(mcp_tools["fabric_list_patterns"], "fn"))
        self._test_get_pattern_details_tool(
            getattr(mcp_tools["fabric_get_pattern_details"], "fn")
        )
        self._test_run_pattern_tool(getattr(mcp_tools["fabric_run_pattern"], "fn"))
        self._test_list_models_tool(getattr(mcp_tools["fabric_list_models"], "fn"))
        self._test_list_strategies_tool(
            getattr(mcp_tools["fabric_list_strategies"], "fn")
        )
        self._test_get_configuration_tool(
            getattr(mcp_tools["fabric_get_configuration"], "fn")
        )

    def _test_list_patterns_tool(
        self, fabric_list_patterns: Callable[..., Any]
    ) -> None:
        # Test fabric_list_patterns with new shared utilities
        builder = FabricApiMockBuilder().with_successful_pattern_list(
            COMMON_PATTERN_LIST
        )
        with mock_fabric_api_client(builder):
            patterns_result: list[str] = fabric_list_patterns()
            assert isinstance(patterns_result, list)
            assert len(patterns_result) == 3

    def _test_get_pattern_details_tool(
        self, fabric_get_pattern_details: Callable[..., Any]
    ) -> None:
        # Test fabric_get_pattern_details with new shared utilities
        builder = FabricApiMockBuilder().with_successful_pattern_details(
            "test_pattern", "Test pattern description", "# Test pattern system prompt"
        )
        with mock_fabric_api_client(builder):
            pattern_details_result: dict[str, str] = fabric_get_pattern_details(
                "test_pattern"
            )
            assert isinstance(pattern_details_result, dict)
            assert "name" in pattern_details_result
            assert "description" in pattern_details_result
            assert "system_prompt" in pattern_details_result
            assert pattern_details_result["name"] == "test_pattern"
            assert pattern_details_result["description"] == "Test pattern description"
            assert (
                pattern_details_result["system_prompt"]
                == "# Test pattern system prompt"
            )

    def _test_run_pattern_tool(self, fabric_run_pattern: Callable[..., Any]) -> None:
        # Test fabric_run_pattern with new shared utilities
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")

        with mock_fabric_api_client(builder) as mock_api_client:
            run_pattern_result = fabric_run_pattern("test_pattern", "test_input")
            assert isinstance(run_pattern_result, dict)
            assert "output_format" in run_pattern_result
            assert "output_text" in run_pattern_result
            assert run_pattern_result["output_text"] == "Hello, World!"
            assert run_pattern_result["output_format"] == "text"
            mock_api_client.close.assert_called_once()

    def _test_list_models_tool(self, fabric_list_models: Callable[..., Any]) -> None:
        # Test fabric_list_models with mocked API
        builder = FabricApiMockBuilder().with_successful_models_list()
        with mock_fabric_api_client(builder):
            models_result: dict[str, Any] = fabric_list_models()
            assert isinstance(models_result, dict)
            assert "models" in models_result
            assert "vendors" in models_result
            assert isinstance(models_result["models"], list)
            assert isinstance(models_result["vendors"], dict)

            # Verify default test data structure
            models = cast(list[str], models_result["models"])
            vendors = cast(dict[str, list[str]], models_result["vendors"])
            assert isinstance(models, list)
            assert isinstance(vendors, dict)
            assert len(models) == 4  # gpt-4o, gpt-3.5-turbo, claude-3-opus, llama2
            assert "gpt-4o" in models
            assert "claude-3-opus" in models

            # Verify vendor structure
            assert "openai" in vendors
            assert "anthropic" in vendors
            assert "ollama" in vendors
            assert "gpt-4o" in vendors["openai"]
            assert "claude-3-opus" in vendors["anthropic"]

    def _test_list_strategies_tool(
        self, fabric_list_strategies: Callable[..., Any]
    ) -> None:
        # Test fabric_list_strategies with mocked API
        builder = FabricApiMockBuilder().with_successful_strategies_list()
        with mock_fabric_api_client(builder):
            strategies_result: dict[str, list[dict[str, str]]] = (
                fabric_list_strategies()
            )
            assert isinstance(strategies_result, dict)
            assert "strategies" in strategies_result
            assert isinstance(strategies_result["strategies"], list)
            strategies_list = strategies_result["strategies"]
            # Verify structure of first strategy
            first_strategy: dict[str, str] = strategies_list[0]
            assert "name" in first_strategy
            assert "description" in first_strategy
            assert "prompt" in first_strategy

    def _test_get_configuration_tool(
        self, fabric_get_configuration: Callable[..., Any]
    ) -> None:
        # Mock the configuration API response
        mock_config_data = {
            "openai_api_key": "sk-abc123def456",
            "anthropic_api_key": "",  # Empty sensitive value
            "ollama_url": "http://localhost:11434",
            "regular_setting": "value123",
        }

        builder = FabricApiMockBuilder().with_json_response(mock_config_data)
        with mock_fabric_api_client(builder):
            config_result: dict[str, Any] = fabric_get_configuration()
            assert isinstance(config_result, dict)
            # Check that we have some configuration data
            assert len(config_result) > 0
            # Check that sensitive values are redacted when present
            for key, value in config_result.items():
                if (
                    isinstance(value, str)
                    and value
                    and any(
                        key.lower().find(pattern.replace("*", "")) != -1
                        for pattern in [
                            "*api_key*",
                            "*token*",
                            "*secret*",
                            "*password*",
                        ]
                    )
                ):
                    # If it's a sensitive key with a value, it should be redacted
                    assert value in ("[REDACTED_BY_MCP_SERVER]", "")

    def test_http_streamable_method_runs_mcp(self, server: FabricMCP):
        """Test that the http_streamable method calls mcp.run() with streamable-http."""
        with patch.object(server, "run") as mock_run:
            # Mock run to avoid actually starting the server
            mock_run.return_value = None

            # Test with default parameters
            server.http_streamable()

            # Verify mcp.run was called with streamable-http transport and defaults
            mock_run.assert_called_once_with(
                transport="streamable-http",
                host="127.0.0.1",
                port=8000,
                path="/message",
            )

    def test_http_streamable_method_with_custom_config(self, server: FabricMCP):
        """Test that the http_streamable method calls mcp.run() with custom config."""
        with patch.object(server, "run") as mock_run:
            # Mock run to avoid actually starting the server
            mock_run.return_value = None

            # Test with custom parameters
            server.http_streamable(host="0.0.0.0", port=9000, mcp_path="/api/mcp")

            # Verify mcp.run was called with streamable-http transport and custom config
            mock_run.assert_called_once_with(
                transport="streamable-http",
                host="0.0.0.0",
                port=9000,
                path="/api/mcp",
            )

    def test_http_streamable_method_handles_keyboard_interrupt(self, server: FabricMCP):
        """Test that the http_streamable method handles KeyboardInterrupt gracefully."""
        with patch.object(server, "run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt

            # Should not raise exception
            server.http_streamable()

            mock_run.assert_called_once()

    def test_http_streamable_method_handles_cancelled_error(self, server: FabricMCP):
        """Test that the http_streamable method handles CancelledError gracefully."""
        with patch.object(server, "run") as mock_run:
            mock_run.side_effect = CancelledError

            # Should not raise exception
            server.http_streamable()

            mock_run.assert_called_once()

    def test_http_streamable_method_handles_would_block(self, server: FabricMCP):
        """Test that the http_streamable method handles WouldBlock gracefully."""
        with patch.object(server, "run") as mock_run:
            mock_run.side_effect = WouldBlock

            # Should not raise exception
            server.http_streamable()

            mock_run.assert_called_once()

    def test_server_initialization_with_default_config_logging(self):
        """Test server initialization logs different default config scenarios."""
        # Test case 1: Both model and vendor present
        with patch(
            "fabric_mcp.core.get_default_model", return_value=("gpt-4", "openai")
        ):
            with patch("fabric_mcp.core.logging") as mock_logging:
                mock_logger = Mock()
                mock_logging.getLogger.return_value = mock_logger

                FabricMCP(log_level="DEBUG")
                mock_logger.info.assert_called_with(
                    "Loaded default model configuration: %s (%s)", "gpt-4", "openai"
                )

        # Test case 2: Only model present
        with patch("fabric_mcp.core.get_default_model", return_value=("gpt-4", None)):
            with patch("fabric_mcp.core.logging") as mock_logging:
                mock_logger = Mock()
                mock_logging.getLogger.return_value = mock_logger

                FabricMCP(log_level="DEBUG")
                mock_logger.info.assert_called_with(
                    "Loaded ONLY default model: %s (no vendor)", "gpt-4"
                )

        # Test case 3: Only vendor present
        with patch("fabric_mcp.core.get_default_model", return_value=(None, "openai")):
            with patch("fabric_mcp.core.logging") as mock_logging:
                mock_logger = Mock()
                mock_logging.getLogger.return_value = mock_logger

                FabricMCP(log_level="DEBUG")
                mock_logger.info.assert_called_with(
                    "Loaded ONLY default vendor: %s (no model)", "openai"
                )

        # Test case 4: Neither present
        with patch("fabric_mcp.core.get_default_model", return_value=(None, None)):
            with patch("fabric_mcp.core.logging") as mock_logging:
                mock_logger = Mock()
                mock_logging.getLogger.return_value = mock_logger

                FabricMCP(log_level="DEBUG")
                mock_logger.info.assert_called_with(
                    "No default model configuration found"
                )

    def test_server_initialization_handles_config_load_error(self):
        """Test server initialization handles errors loading default config."""
        error = OSError("File not found")
        with patch("fabric_mcp.core.get_default_model", side_effect=error):
            with patch("fabric_mcp.core.logging") as mock_logging:
                mock_logger = Mock()
                mock_logging.getLogger.return_value = mock_logger

                server = FabricMCP(log_level="DEBUG")

                mock_logger.warning.assert_called_with(
                    "Failed to load default model configuration: %s. "
                    "Pattern execution will use hardcoded defaults.",
                    error,
                )
            # Verify server still initializes with None defaults
            assert server.get_default_model_config() == (None, None)

    def test_get_default_model_config_method(self):
        """Test the public getter for default model configuration."""
        with patch(
            "fabric_mcp.core.get_default_model", return_value=("claude-3", "anthropic")
        ):
            server = FabricMCP()
            model, vendor = server.get_default_model_config()
            assert model == "claude-3"
            assert vendor == "anthropic"


class TestFabricGetConfiguration(TestFixturesBase):
    """Test fabric_get_configuration tool implementation and redaction logic."""

    def test_fabric_get_configuration_successful_call(self, server: FabricMCP):
        """Test successful API call with mixed sensitive/non-sensitive config."""
        mock_config_data = {
            "openai_api_key": "sk-abc123def456",
            "anthropic_api_key": "ant-abc123",
            "google_api_key": "",  # Empty sensitive value
            "ollama_url": "http://localhost:11434",
            "fabric_config_dir": "~/.config/fabric",
            "default_model": "gpt-4",
            "custom_token": "secret123",
            "user_password": "mypassword",
            "empty_secret": "",
            "regular_setting": "value123",
        }

        builder = FabricApiMockBuilder().with_json_response(mock_config_data)

        with mock_fabric_api_client(builder) as mock_api:
            result = server.fabric_get_configuration()

            # Verify API was called correctly
            mock_api.get.assert_called_once_with("/config")
            mock_api.close.assert_called_once()

            # Verify result structure
            assert isinstance(result, dict)

            # Verify redaction of sensitive keys with values
            assert result["openai_api_key"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["anthropic_api_key"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["custom_token"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["user_password"] == "[REDACTED_BY_MCP_SERVER]"

            # Verify empty sensitive values are passed through
            assert result["google_api_key"] == ""
            assert result["empty_secret"] == ""

            # Verify non-sensitive values are passed through
            assert result["ollama_url"] == "http://localhost:11434"
            assert result["fabric_config_dir"] == "~/.config/fabric"
            assert result["default_model"] == "gpt-4"
            assert result["regular_setting"] == "value123"

    def test_fabric_get_configuration_redaction_patterns(self, server: FabricMCP):
        """Test redaction patterns work with various key formats."""
        mock_config_data = {
            "OPENAI_API_KEY": "value1",  # Uppercase
            "stripe_api_key": "value2",  # Lowercase
            "Custom_API_Key": "value3",  # Mixed case
            "AUTH_TOKEN": "value4",
            "refresh_token": "value5",
            "DB_SECRET": "value6",
            "app_secret": "value7",
            "USER_PASSWORD": "value8",
            "admin_password": "value9",
            "some_setting": "value10",  # Non-sensitive
        }

        builder = FabricApiMockBuilder().with_json_response(mock_config_data)

        with mock_fabric_api_client(builder):
            result = server.fabric_get_configuration()

            # Verify all sensitive patterns are redacted
            assert result["OPENAI_API_KEY"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["stripe_api_key"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["Custom_API_Key"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["AUTH_TOKEN"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["refresh_token"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["DB_SECRET"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["app_secret"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["USER_PASSWORD"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["admin_password"] == "[REDACTED_BY_MCP_SERVER]"

            # Verify non-sensitive value is passed through
            assert result["some_setting"] == "value10"

    def test_fabric_get_configuration_empty_sensitive_values(self, server: FabricMCP):
        """Test that empty sensitive values are passed through, not redacted."""
        mock_config_data = {
            "openai_api_key": "",
            "auth_token": "",
            "db_secret": "",
            "user_password": "",
            "regular_setting": "",
        }

        builder = FabricApiMockBuilder().with_json_response(mock_config_data)

        with mock_fabric_api_client(builder):
            result = server.fabric_get_configuration()

            # Verify empty sensitive values are NOT redacted
            assert result["openai_api_key"] == ""
            assert result["auth_token"] == ""
            assert result["db_secret"] == ""
            assert result["user_password"] == ""
            assert result["regular_setting"] == ""

    def test_fabric_get_configuration_api_connection_error(self, server: FabricMCP):
        """Test handling of API connection errors."""
        builder = FabricApiMockBuilder().with_connection_error()

        with mock_fabric_api_client(builder):
            with pytest.raises(Exception) as exc_info:
                server.fabric_get_configuration()

            # Should raise McpError with appropriate message
            assert "Failed to connect to Fabric API" in str(exc_info.value)

    def test_fabric_get_configuration_http_status_error(self, server: FabricMCP):
        """Test handling of HTTP status errors (4xx, 5xx responses)."""
        builder = FabricApiMockBuilder().with_http_error(500, "Internal Server Error")

        with mock_fabric_api_client(builder):
            with pytest.raises(Exception) as exc_info:
                server.fabric_get_configuration()

            # Should raise McpError with appropriate message
            assert "Fabric API error during retrieving configuration: 500" in str(
                exc_info.value
            )

    def test_fabric_get_configuration_invalid_response_type(self, server: FabricMCP):
        """Test handling of invalid JSON response (non-dict)."""
        # Mock API returning a list instead of dict
        builder = FabricApiMockBuilder().with_raw_response_data(["not", "a", "dict"])

        with mock_fabric_api_client(builder):
            with pytest.raises(Exception) as exc_info:
                server.fabric_get_configuration()

            # Should raise McpError about invalid response type
            assert "expected dict for config" in str(exc_info.value)

    def test_fabric_get_configuration_api_key_value_redaction(self, server: FabricMCP):
        """Test redaction of values that look like API keys regardless of key name."""
        mock_config_data = {
            # Real Fabric config format: vendor names as keys, API keys as values
            "openai": "sk-1234567890abcdef",  # OpenAI API key prefix
            "anthropic": "ant-api-key-12345",  # Anthropic API key prefix
            "deepseek": "sk-deepseek123456",  # DeepSeek uses sk- prefix
            "groq": "gsk_1234567890",  # Groq API key prefix
            "grokai": "xai-grokai123456",  # Grokai API key prefix
            "gemini": "AIzaSyABC123456",  # Google API key prefix
            "openrouter": "sk-or-v1-1234567890",  # OpenRouter API key
            "lmstudio": "",  # Empty value should pass through
            "ollama": "",  # Empty value should pass through
            "fabric_config_dir": "~/.config/fabric",  # Non-API key value
            "debug_mode": True,  # Non-string value
        }

        builder = FabricApiMockBuilder().with_json_response(mock_config_data)

        with mock_fabric_api_client(builder):
            result = server.fabric_get_configuration()

            # Verify API key values are redacted regardless of key name
            assert result["openai"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["anthropic"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["deepseek"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["groq"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["grokai"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["gemini"] == "[REDACTED_BY_MCP_SERVER]"
            assert result["openrouter"] == "[REDACTED_BY_MCP_SERVER]"

            # Verify empty values are passed through
            assert result["lmstudio"] == ""
            assert result["ollama"] == ""

            # Verify non-API key values are passed through
            assert result["fabric_config_dir"] == "~/.config/fabric"
            assert result["debug_mode"] is True
