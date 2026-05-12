"""Tests for exposing Fabric patterns as MCP prompts."""

import pytest
from fastmcp.exceptions import NotFoundError, PromptError
from mcp.types import TextContent

from fabric_mcp.core import FabricMCP
from tests.shared.fabric_api_mocks import FabricApiMockBuilder, mock_fabric_api_client


@pytest.mark.asyncio
async def test_fabric_patterns_are_discoverable_as_mcp_prompts() -> None:
    """Fabric patterns should be exposed through MCP prompts/list."""
    server = FabricMCP()
    builder = FabricApiMockBuilder().with_successful_pattern_list(
        ["analyze_claims", "create-story"]
    )

    with mock_fabric_api_client(builder) as mock_client:
        prompts = await server.list_prompts()

    assert [prompt.name for prompt in prompts] == ["analyze_claims", "create-story"]
    assert prompts[0].title == "Analyze Claims"
    assert prompts[0].description == "Fabric pattern prompt"
    assert prompts[0].arguments is not None
    assert prompts[0].arguments[0].name == "input_text"
    mock_client.get.assert_called_once_with("/patterns/names")
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_fabric_pattern_prompt_renders_pattern_details() -> None:
    """prompts/get should return a user message containing the pattern text."""
    server = FabricMCP()
    builder = FabricApiMockBuilder().with_successful_pattern_details(
        name="analyze_claims",
        description="Analyze truth claims",
        system_prompt="# IDENTITY\nYou are an expert fact checker.",
    )

    with mock_fabric_api_client(builder) as mock_client:
        result = await server.render_prompt(
            "analyze_claims", {"input_text": "This claim needs review."}
        )

    assert result.description == "Analyze truth claims"
    assert len(result.messages) == 1
    assert result.messages[0].role == "user"
    content = result.messages[0].content
    assert isinstance(content, TextContent)
    assert content.text == (
        "# IDENTITY\nYou are an expert fact checker.\n\nThis claim needs review."
    )
    mock_client.get.assert_called_once_with("/patterns/analyze_claims")
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_fabric_pattern_prompt_rejects_unknown_arguments() -> None:
    """Prompt rendering should validate arguments before returning messages."""
    server = FabricMCP()
    builder = FabricApiMockBuilder().with_successful_pattern_details(
        name="summarize",
        description="Summarize text",
        system_prompt="Summarize the provided text.",
    )

    with mock_fabric_api_client(builder):
        with pytest.raises(PromptError, match="Unexpected prompt argument"):
            await server.render_prompt("summarize", {"unexpected": "value"})


@pytest.mark.asyncio
async def test_unknown_fabric_pattern_prompt_is_not_found() -> None:
    """An unknown Fabric pattern should not render as an MCP prompt."""
    server = FabricMCP()
    builder = FabricApiMockBuilder().with_http_error(
        status_code=500,
        response_text="open /patterns/missing: no such file or directory",
    )

    with mock_fabric_api_client(builder):
        with pytest.raises(NotFoundError, match="Unknown prompt"):
            await server.render_prompt("missing", {})
