"""MCP prompt support for Fabric patterns."""

import logging
from typing import Any

from fastmcp.prompts import Message, Prompt, PromptArgument, PromptResult
from fastmcp.server.providers.base import Provider
from mcp.shared.exceptions import McpError

from .fabric_tools import FabricToolsMixin

INPUT_TEXT_ARGUMENT_DESCRIPTION = (
    "Optional text to append after the Fabric pattern instructions."
)


class FabricPatternPrompt(Prompt):
    """A Fabric pattern exposed as an MCP prompt template."""

    system_prompt: str

    async def render(self, arguments: dict[str, Any] | None = None) -> PromptResult:
        """Render a Fabric pattern as MCP prompt messages."""
        input_text = ""
        if arguments:
            unexpected_arguments = sorted(set(arguments) - {"input_text"})
            if unexpected_arguments:
                unexpected = ", ".join(unexpected_arguments)
                raise ValueError(f"Unexpected prompt argument(s): {unexpected}")

            raw_input_text = arguments.get("input_text", "")
            if raw_input_text is None:
                raw_input_text = ""
            if not isinstance(raw_input_text, str):
                raise TypeError("input_text must be a string")
            input_text = raw_input_text

        prompt_text = self.system_prompt
        if input_text:
            prompt_text = f"{prompt_text}\n\n{input_text}"

        return PromptResult(
            messages=[Message(prompt_text, role="user")],
            description=self.description,
            meta=self.meta,
        )


class FabricPatternPromptProvider(Provider):
    """Dynamic prompt provider that exposes Fabric patterns as MCP prompts."""

    def __init__(self, fabric_tools: FabricToolsMixin) -> None:
        """Create a provider backed by Fabric pattern API methods."""
        super().__init__()
        self._fabric_tools = fabric_tools
        self._logger = logging.getLogger(__name__)

    async def _list_prompts(self) -> list[Prompt]:
        """List Fabric patterns as MCP prompts."""
        try:
            pattern_names = self._fabric_tools.fabric_list_patterns()
        except (McpError, ValueError, TypeError) as e:
            self._logger.warning("Unable to list Fabric pattern prompts: %s", e)
            return []

        return [
            FabricPatternPrompt(
                name=pattern_name,
                title=_pattern_title(pattern_name),
                description="Fabric pattern prompt",
                system_prompt="",
                arguments=[
                    PromptArgument(
                        name="input_text",
                        description=INPUT_TEXT_ARGUMENT_DESCRIPTION,
                        required=False,
                    )
                ],
            )
            for pattern_name in pattern_names
        ]

    async def _get_prompt(self, name: str, version: Any | None = None) -> Prompt | None:
        """Get a Fabric pattern prompt by name."""
        del version  # Fabric patterns are currently not versioned through the API.
        try:
            details = self._fabric_tools.fabric_get_pattern_details(name)
        except (McpError, ValueError, TypeError) as e:
            self._logger.debug("Unable to get Fabric pattern prompt %r: %s", name, e)
            return None

        return FabricPatternPrompt(
            name=details["name"],
            title=_pattern_title(details["name"]),
            description=details["description"] or None,
            system_prompt=details["system_prompt"],
            arguments=[
                PromptArgument(
                    name="input_text",
                    description=INPUT_TEXT_ARGUMENT_DESCRIPTION,
                    required=False,
                )
            ],
        )


def _pattern_title(pattern_name: str) -> str:
    """Convert a Fabric pattern identifier into a display title."""
    return pattern_name.replace("_", " ").replace("-", " ").title()
