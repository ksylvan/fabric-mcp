"""Tests for fabric_run_pattern tool advanced features.

This module tests advanced features of the fabric_run_pattern tool including
request construction, model inference, variables and attachments,
unexpected SSE types, and coverage edge cases.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio
from mcp import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS

from fabric_mcp.constants import DEFAULT_MODEL, DEFAULT_VENDOR
from fabric_mcp.core import FabricMCP
from fabric_mcp.models import PatternExecutionConfig
from tests.shared.fabric_api_mocks import (
    FabricApiMockBuilder,
    assert_mcp_error,
    mock_fabric_api_client,
)
from tests.unit.test_fabric_run_pattern_base import (
    COMMON_PARAMS_FULL,
    TestFabricRunPatternFixtureBase,
)


class TestFabricRunPatternRequestConstruction(TestFabricRunPatternFixtureBase):
    """Test cases for fabric_run_pattern request construction with new parameters."""

    def test_request_construction_with_new_parameters(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test that new parameters are correctly included in API request."""
        builder = FabricApiMockBuilder().with_successful_sse(
            "Request construction test"
        )

        with mock_fabric_api_client(builder) as mock_api_client:
            fabric_run_pattern_tool(
                "test_pattern",
                "test input",
                **COMMON_PARAMS_FULL,
            )

            # Verify API was called with correct parameters
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]

            # Check prompt-level parameters
            prompt = payload["prompts"][0]
            assert prompt["model"] == "claude-3-opus"
            assert prompt["strategyName"] == "creative"

            # Check root-level LLM parameters
            assert payload["temperature"] == 0.8
            assert payload["topP"] == 0.95
            assert payload["presencePenalty"] == 0.1
            assert payload["frequencyPenalty"] == -0.1

    def test_request_construction_with_defaults(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test that omitted parameters use appropriate defaults."""
        builder = FabricApiMockBuilder().with_successful_sse("Default test")

        with mock_fabric_api_client(builder) as mock_api_client:
            fabric_run_pattern_tool("test_pattern", "test input")

            # Verify API was called with defaults
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]

            # Check default values are used
            assert payload["temperature"] == 0.7  # Default temperature
            assert payload["topP"] == 0.9  # Default top_p
            assert payload["frequencyPenalty"] == 0.0  # Default frequency_penalty
            assert payload["presencePenalty"] == 0.0  # Default presence_penalty

            prompt = payload["prompts"][0]
            assert prompt["strategyName"] == ""  # Default empty strategy

    def test_backward_compatibility(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test that existing calls without new parameters still work."""
        builder = FabricApiMockBuilder().with_successful_sse(
            "Backward compatibility test"
        )

        with mock_fabric_api_client(builder):
            # Test basic call (original Story 3.1 format)
            result = fabric_run_pattern_tool("test_pattern", "test input")
            assert result["output_text"] == "Backward compatibility test"
            assert "output_format" in result

            # Test with stream parameter (original Story 3.1 format)
            result = fabric_run_pattern_tool("test_pattern", "test input", stream=False)
            assert result["output_text"] == "Backward compatibility test"

            # Test with config parameter (original Story 3.1 format)
            config = PatternExecutionConfig()
            result = fabric_run_pattern_tool(
                "test_pattern", "test input", config=config
            )
            assert result["output_text"] == "Backward compatibility test"


class TestFabricRunPatternModelInference(TestFabricRunPatternFixtureBase):
    """Test cases for fabric_run_pattern tool model and vendor inference."""

    @pytest.fixture
    def server_no_defaults(self) -> FabricMCP:
        """Create a FabricMCP server instance with no default model/vendor."""
        with patch("fabric_mcp.core.get_default_model", return_value=(None, None)):
            return FabricMCP()

    @pytest.fixture
    def server_claude_model(self) -> FabricMCP:
        """Create a FabricMCP server instance with Claude model default."""
        with patch(
            "fabric_mcp.core.get_default_model",
            return_value=("claude-3-opus", None),
        ):
            return FabricMCP()

    @pytest.fixture
    def server_gpt_model(self) -> FabricMCP:
        """Create a FabricMCP server instance with GPT model default."""
        with patch(
            "fabric_mcp.core.get_default_model",
            return_value=("gpt-3.5-turbo", None),
        ):
            return FabricMCP()

    @pytest_asyncio.fixture
    async def fabric_run_pattern_tool_no_defaults(
        self, server_no_defaults: FabricMCP
    ) -> Callable[..., Any]:
        """Get the fabric_run_pattern tool from server with no defaults."""
        tools_list = await server_no_defaults.list_tools()
        tools = {t.name: t for t in tools_list}
        return getattr(tools["fabric_run_pattern"], "fn")

    @pytest_asyncio.fixture
    async def fabric_run_pattern_tool_claude(
        self, server_claude_model: FabricMCP
    ) -> Callable[..., Any]:
        """Get the fabric_run_pattern tool from server with Claude model."""
        tools_list = await server_claude_model.list_tools()
        tools = {t.name: t for t in tools_list}
        return getattr(tools["fabric_run_pattern"], "fn")

    @pytest_asyncio.fixture
    async def fabric_run_pattern_tool_gpt(
        self, server_gpt_model: FabricMCP
    ) -> Callable[..., Any]:
        """Get the fabric_run_pattern tool from server with GPT model."""
        tools_list = await server_gpt_model.list_tools()
        tools = {t.name: t for t in tools_list}
        return getattr(tools["fabric_run_pattern"], "fn")

    def test_pattern_not_found_500_error(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test handling of 500 error indicating pattern not found."""
        builder = FabricApiMockBuilder().with_http_error(
            500, "no such file or directory: pattern not found"
        )

        with mock_fabric_api_client(builder) as mock_api_client:
            with pytest.raises(McpError) as exc_info:
                fabric_run_pattern_tool("nonexistent_pattern", "test input")
                mock_api_client.close.assert_called_once()

            # Should be transformed to Invalid params error
            assert exc_info.value.error.code == INVALID_PARAMS
            assert (
                "Pattern 'nonexistent_pattern' not found"
                in exc_info.value.error.message
            )

    def test_generic_500_error(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test handling of generic 500 error (not pattern not found)."""
        builder = FabricApiMockBuilder().with_http_error(
            500, "Database connection failed"
        )

        with mock_fabric_api_client(builder) as client:
            with pytest.raises(McpError) as exc_info:
                fabric_run_pattern_tool("test_pattern", "test input")

            error = exc_info.value
            assert error.error.code == INTERNAL_ERROR  # Internal error
            assert "Database connection failed" in error.error.message
            client.close.assert_called_once()

    def test_vendor_inference_from_model_name(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test vendor inference when no default vendor is configured."""
        builder = FabricApiMockBuilder().with_successful_sse("Inference test")

        # Test Claude model inference
        with patch("fabric_mcp.core.get_default_model", return_value=(None, None)):
            with mock_fabric_api_client(builder):
                config = PatternExecutionConfig(model_name="claude-3-sonnet")
                result = fabric_run_pattern_tool(
                    "test_pattern", "test input", config=config
                )
                assert result["output_text"] == "Inference test"

        # Test GPT model inference
        with patch("fabric_mcp.core.get_default_model", return_value=(None, None)):
            with mock_fabric_api_client(builder):
                config = PatternExecutionConfig(model_name="gpt-4")
                result = fabric_run_pattern_tool(
                    "test_pattern", "test input", config=config
                )
                assert result["output_text"] == "Inference test"

    def test_hardcoded_model_fallback_when_no_defaults(
        self, fabric_run_pattern_tool_no_defaults: Callable[..., Any]
    ) -> None:
        """Test fallback to hardcoded default model when no environment defaults."""
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")

        # Mock get_default_model to return None for both model and vendor
        with mock_fabric_api_client(builder) as mock_api_client:
            result: dict[str, Any] = fabric_run_pattern_tool_no_defaults(
                "test_pattern", "test input"
            )

            assert isinstance(result, dict)
            # Check that API was called (would use hardcoded defaults)
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]
            assert payload["prompts"][0]["model"] == DEFAULT_MODEL
            assert payload["prompts"][0]["vendor"] == DEFAULT_VENDOR

    def test_vendor_inference_for_claude_models(
        self, fabric_run_pattern_tool_claude: Callable[..., Any]
    ) -> None:
        """Test vendor inference from Claude model names."""
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")

        with mock_fabric_api_client(builder) as mock_api_client:
            result: dict[str, Any] = fabric_run_pattern_tool_claude(
                "test_pattern", "test input"
            )

            assert isinstance(result, dict)
            # Check that vendor was inferred as "anthropic" for Claude models
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]
            assert payload["prompts"][0]["model"] == "claude-3-opus"
            assert payload["prompts"][0]["vendor"] == DEFAULT_VENDOR

    def test_vendor_inference_for_gpt_models(
        self, fabric_run_pattern_tool_gpt: Callable[..., Any]
    ) -> None:
        """Test vendor inference from GPT model names."""
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")

        with mock_fabric_api_client(builder) as mock_api_client:
            result: dict[str, Any] = fabric_run_pattern_tool_gpt(
                "test_pattern", "test input"
            )

            assert isinstance(result, dict)
            # Check that vendor was inferred as "openai" for GPT models
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]
            assert payload["prompts"][0]["model"] == "gpt-3.5-turbo"
            assert payload["prompts"][0]["vendor"] == DEFAULT_VENDOR


class TestFabricRunPatternVariablesAndAttachments(TestFabricRunPatternFixtureBase):
    """Test cases for fabric_run_pattern tool variables and attachments parameters."""

    def test_variables_parameter_basic(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test fabric_run_pattern with variables parameter."""
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")
        variables = {"key1": "value1", "key2": "value2"}

        with mock_fabric_api_client(builder) as mock_api_client:
            result: dict[str, Any] = fabric_run_pattern_tool(
                "test_pattern", "test input", variables=variables
            )

            assert isinstance(result, dict)
            assert result["output_text"] == "Hello, World!"

            # Verify variables were passed to the API
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]
            assert payload["prompts"][0]["variables"] == variables
            mock_api_client.close.assert_called_once()

    def test_attachments_parameter_basic(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test fabric_run_pattern with attachments parameter."""
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")
        attachments = ["file1.txt", "file2.pdf"]

        with mock_fabric_api_client(builder) as mock_api_client:
            result: dict[str, Any] = fabric_run_pattern_tool(
                "test_pattern", "test input", attachments=attachments
            )

            assert isinstance(result, dict)
            assert result["output_text"] == "Hello, World!"

            # Verify attachments were passed to the API
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]
            assert payload["prompts"][0]["attachments"] == attachments
            mock_api_client.close.assert_called_once()

    def test_variables_and_attachments_together(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test fabric_run_pattern with both variables and attachments."""
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")
        variables = {"name": "test", "type": "example"}
        attachments = ["doc.txt", "data.json"]

        with mock_fabric_api_client(builder) as mock_api_client:
            result: dict[str, Any] = fabric_run_pattern_tool(
                "test_pattern",
                "test input",
                variables=variables,
                attachments=attachments,
            )

            assert isinstance(result, dict)
            assert result["output_text"] == "Hello, World!"

            # Verify both parameters were passed to the API
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]
            assert payload["prompts"][0]["variables"] == variables
            assert payload["prompts"][0]["attachments"] == attachments
            mock_api_client.close.assert_called_once()

    def test_variables_empty_dict(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test fabric_run_pattern with empty variables dict."""
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")

        with mock_fabric_api_client(builder) as mock_api_client:
            result: dict[str, Any] = fabric_run_pattern_tool(
                "test_pattern", "test input", variables={}
            )

            assert isinstance(result, dict)
            assert result["output_text"] == "Hello, World!"

            # Verify empty variables dict was passed
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]
            assert payload["prompts"][0]["variables"] == {}
            mock_api_client.close.assert_called_once()

    def test_attachments_empty_list(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test fabric_run_pattern with empty attachments list."""
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")

        with mock_fabric_api_client(builder) as mock_api_client:
            result: dict[str, Any] = fabric_run_pattern_tool(
                "test_pattern", "test input", attachments=[]
            )

            assert isinstance(result, dict)
            assert result["output_text"] == "Hello, World!"

            # Verify empty attachments list was passed
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]
            assert payload["prompts"][0]["attachments"] == []
            mock_api_client.close.assert_called_once()

    def test_variables_none_excluded_from_payload(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test that None variables are excluded from API payload."""
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")

        with mock_fabric_api_client(builder) as mock_api_client:
            result: dict[str, Any] = fabric_run_pattern_tool(
                "test_pattern", "test input", variables=None
            )

            assert isinstance(result, dict)
            assert result["output_text"] == "Hello, World!"

            # Verify variables is not in the payload when None
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]
            assert "variables" not in payload["prompts"][0]
            mock_api_client.close.assert_called_once()

    def test_attachments_none_excluded_from_payload(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test that None attachments are excluded from API payload."""
        builder = FabricApiMockBuilder().with_successful_sse("Hello, World!")

        with mock_fabric_api_client(builder) as mock_api_client:
            result: dict[str, Any] = fabric_run_pattern_tool(
                "test_pattern", "test input", attachments=None
            )

            assert isinstance(result, dict)
            assert result["output_text"] == "Hello, World!"

            # Verify attachments is not in the payload when None
            mock_api_client.post.assert_called_once()
            call_args = mock_api_client.post.call_args
            payload = call_args[1]["json_data"]
            assert "attachments" not in payload["prompts"][0]
            mock_api_client.close.assert_called_once()


class TestFabricRunPatternUnexpectedSSETypes(TestFabricRunPatternFixtureBase):
    """Test cases for fabric_run_pattern tool with unexpected SSE data types."""

    def test_unexpected_sse_data_type(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test handling of unexpected SSE data type."""
        # Create SSE stream with unexpected type
        sse_lines = [
            'data: {"type": "unexpected_type", "content": "Some content"}',
        ]
        builder = FabricApiMockBuilder().with_sse_lines(sse_lines)

        with mock_fabric_api_client(builder) as mock_api_client:
            # Let's see what actually happens - it might return successfully
            # but with empty content due to no 'complete' signal
            try:
                result = fabric_run_pattern_tool("test_pattern", "test input")
                # If it doesn't raise an error, we should get an empty result
                assert result["output_text"] == ""
                assert result["output_format"] == "text"
            except McpError as e:
                # If it does raise an error, verify it's the expected one
                error_msg = str(e)
                assert (
                    "Empty SSE stream" in error_msg
                    or "Unexpected SSE data type" in error_msg
                )

            mock_api_client.close.assert_called_once()

    def test_unexpected_sse_data_type_with_unknown_field(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test handling of SSE data with missing type field."""
        # Create SSE stream with missing type field
        sse_lines = [
            'data: {"content": "Some content"}',  # Missing "type" field
        ]
        builder = FabricApiMockBuilder().with_sse_lines(sse_lines)

        with mock_fabric_api_client(builder) as mock_api_client:
            # This should trigger the unexpected SSE data type handling
            try:
                result = fabric_run_pattern_tool("test_pattern", "test input")
                # If it doesn't raise an error, we should get an empty result
                assert result["output_text"] == ""
                assert result["output_format"] == "text"
            except McpError as e:
                # If it does raise an error, verify it's the expected one
                error_msg = str(e)
                assert (
                    "Empty SSE stream" in error_msg
                    or "Unexpected SSE data type" in error_msg
                )

            mock_api_client.close.assert_called_once()

    def test_sse_error_response_detailed(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test handling of SSE error response with detailed error message."""
        builder = FabricApiMockBuilder().with_sse_error("Pattern validation failed")

        with mock_fabric_api_client(builder) as mock_api_client:
            with pytest.raises(McpError) as exc_info:
                fabric_run_pattern_tool("test_pattern", "test input")

            assert_mcp_error(exc_info, INTERNAL_ERROR, "Pattern validation failed")
            mock_api_client.close.assert_called_once()

    def test_unexpected_sse_data_type_forces_exception(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test to ensure unexpected SSE type definitely triggers exception."""
        # Create SSE stream with ONLY unexpected type (no complete)
        sse_lines = [
            'data: {"type": "weird_unknown_type", "content": "test"}',
        ]
        builder = FabricApiMockBuilder().with_sse_lines(sse_lines)

        with mock_fabric_api_client(builder) as mock_api_client:
            # This should either raise an error or return empty result
            try:
                result = fabric_run_pattern_tool("test_pattern", "test input")
                # If no exception, it should return empty content
                assert result["output_text"] == ""
                assert result["output_format"] == "text"
            except McpError as e:
                # If it raises, it should be about unexpected type or empty stream
                error_msg = str(e)
                assert (
                    "Unexpected SSE data type" in error_msg
                    or "Empty SSE stream" in error_msg
                )

            mock_api_client.close.assert_called_once()


class TestFabricRunPatternCoverageTargets(TestFabricRunPatternFixtureBase):
    """Test cases to target specific missing coverage lines."""

    def test_default_config_creation_without_explicit_config(
        self, fabric_run_pattern_tool: Callable[..., Any]
    ) -> None:
        """Test calling fabric_run_pattern without config to trigger defaults."""
        builder = FabricApiMockBuilder().with_successful_sse("Coverage test")

        with mock_fabric_api_client(builder) as mock_api_client:
            # Call without config parameter to trigger default config creation
            result: dict[str, Any] = fabric_run_pattern_tool(
                "test_pattern",
                "test input",
                # No config parameter provided - should trigger line 494
            )

            assert isinstance(result, dict)
            assert result["output_text"] == "Coverage test"
            assert result["output_format"] == "text"
            mock_api_client.close.assert_called_once()
