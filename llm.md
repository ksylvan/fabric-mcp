# llm.txt — Fabric & Fabric MCP Server Usage Guide

## What is Fabric?

Fabric is an open-source AI framework focused on prompt engineering and modular AI workflows. It lets you define, organize, and run “patterns” (specialized prompts and workflows) for tasks like code explanation, summarization, creative writing, and more.

- Patterns are stored as Markdown files (see `/home/gaem/fabric/configs/patterns/`).
- Fabric can be run as a REST API server with `fabric --serve`.

## What is the Fabric MCP Server?

The Fabric MCP Server bridges Fabric’s REST API to any Model Context Protocol (MCP) compatible application (e.g., IDEs, chatbots, LLMs).

- It exposes Fabric’s patterns and capabilities as MCP tools.
- It translates MCP tool calls into Fabric API calls and returns results.

## Key MCP Tools

- `fabric_list_patterns`: Lists available pattern names.
- `fabric_pattern_prompt`: Returns the full prompt for a given pattern.
- `fabric_run_pattern`: (When implemented) Runs a pattern with input and returns the result.

## How to Use

1. **Start Fabric Backend:**
   ```bash
   fabric --serve
   ```
   (Default REST API at http://127.0.0.1:8080)

2. **Start Fabric MCP Server:**
   ```bash
   /home/gaem/fabric/fabric-mcp/.venv/bin/fabric-mcp --stdio
   ```
   (Configured via `/home/gaem/.codeium/windsurf-next/mcp_config.json`)

3. **MCP Client/LLM Usage:**
   - Discover available tools via MCP.
   - Use `fabric_list_patterns` to see all patterns.
   - Use `fabric_pattern_prompt` to get the full Markdown prompt for a pattern.
   - Use `fabric_run_pattern` (when implemented) to run a pattern with input.

## Example Pattern Structure

A pattern prompt (e.g., `ai`) starts with:
```
# IDENTITY and PURPOSE

<one-line summary of what the pattern does>
...
```

## Environment Variables

- `FABRIC_BASE_URL` (default: http://127.0.0.1:8080)
- `FABRIC_API_KEY` (if required by backend)
- `FABRIC_MCP_LOG_LEVEL` (optional)

## Directory Structure

- `/home/gaem/fabric/configs/patterns/` — All pattern definitions (Markdown).
- `/home/gaem/fabric/fabric-mcp/` — MCP server source and config.

---

Let me know if you want a more detailed or more concise version, or if you want to include usage examples!
