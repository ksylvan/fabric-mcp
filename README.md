# Fabric MCP Server

[![License: MIT][mit_license]][mit_license_link]
[![image](https://img.shields.io/pypi/v/fabric-mcp.svg)](https://pypi.python.org/pypi/fabric-mcp)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ksylvan/fabric-mcp)
[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/fca21713-c65c-42ee-b8fa-ce209466e70a)

| main  |   |  | develop  |   |
|:---:|:---:|:---:|:---:|:---:|
| [![Main Tests][main_tests]][main_tests_link] | [![Main Publish][main_publish]][main_publish_link] |  | [![Develop Tests][develop_tests]][develop_tests_link] | [![Develop Publish][develop_publish]][develop_publish_link] |

<div align="center">
<img src="https://github.com/ksylvan/fabric-mcp/blob/main/docs/logo.png?raw=true" alt="fabric-mcp logo" width="200" height="200">
<a href="https://mseep.ai/app/ksylvan-fabric-mcp"><img src="https://mseep.net/pr/ksylvan-fabric-mcp-badge.png" alt="MseeP.ai Security Assessment Badge" height="200"></a>
</div>

Connect the power of the Fabric AI framework to any Model Context Protocol (MCP) compatible application.

This project implements a standalone server that bridges the gap between [Daniel Miessler's Fabric framework][fabricGithubLink] and the [Model Context Protocol (MCP)][MCP]. It allows you to use Fabric's patterns, models, and configurations directly within MCP-enabled environments like IDE extensions or chat interfaces.

Imagine seamlessly using Fabric's specialized prompts for code explanation, refactoring, or creative writing right inside your favorite tools!

## Table of Contents

- [Fabric MCP Server](#fabric-mcp-server)
  - [Table of Contents](#table-of-contents)
  - [What is this?](#what-is-this)
  - [Key Goals \& Features (Based on Design)](#key-goals--features-based-on-design)
  - [How it Works](#how-it-works)
  - [Project Status](#project-status)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation From Source (for developers)](#installation-from-source-for-developers)
    - [Installation From PyPI (for users)](#installation-from-pypi-for-users)
  - [Configuration (Environment Variables)](#configuration-environment-variables)
    - [Transport Options](#transport-options)
  - [Contributing](#contributing)
  - [License](#license)

## What is this?

- **Fabric:** An open-source framework for augmenting human capabilities using AI, focusing on prompt engineering and modular AI workflows.
- **MCP:** An open standard protocol enabling AI applications (like IDEs) to securely interact with external tools and data sources (like this server).
- **Fabric MCP Server:** This project acts as an MCP server, translating MCP requests into calls to a running Fabric instance's REST API (`fabric --serve`).

## Key Goals & Features (Based on Design)

- **Seamless Integration:** Use Fabric patterns and capabilities directly within MCP clients without switching context.
- **Enhanced Workflows:** Empower LLMs within IDEs or other tools to leverage Fabric's specialized prompts and user configurations.
- **Standardization:** Adhere to the open MCP standard for AI tool integration.
- **Leverage Fabric Core:** Build upon the existing Fabric CLI and REST API without modifying the core Fabric codebase.
- **Expose Fabric Functionality:** Provide MCP tools to list patterns, get pattern details, run patterns, list models/strategies, and retrieve configuration.

## How it Works

1. An **MCP Host** (e.g., an IDE extension) connects to this **Fabric MCP Server**.
2. The Host discovers available tools (like `fabric_run_pattern`) via MCP's `list_tools()` mechanism.
3. When the user invokes a tool (e.g., asking the IDE's AI assistant to refactor code using a Fabric pattern), the Host sends an MCP request to this server.
4. The **Fabric MCP Server** translates the MCP request into a corresponding REST API call to a running `fabric --serve` instance.
5. The `fabric --serve` instance executes the pattern.
6. The **Fabric MCP Server** receives the response (potentially streaming) from Fabric and translates it back into an MCP response for the Host.

## Project Status

This project is feature-complete.

The project was completed by using the [BMAD-METHOD (Breakthrough Method of Agile Ai-Driven Development)][bmad-method].

The core architecture and proposed tools are outlined in the [High-Level Architecture Document][architecture_doc].

You can also use [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ksylvan/fabric-mcp) to explore the source code.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (Python package and environment manager) for developers

### Installation From Source (for developers)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ksylvan/fabric-mcp.git
   cd fabric-mcp
   ```

2. **Install dependencies using uv sync:**

   ```bash
   uv sync --dev
   ```

   This command ensures your virtual environment matches the dependencies in `pyproject.toml` and `uv.lock`, creating the environment on the first run if necessary.

3. **Activate the virtual environment (uv will create it if needed):**

   - On macOS/Linux:

     ```bash
     source .venv/bin/activate
     ```

   - On Windows:

     ```bash
     .venv\Scripts\activate
     ```

Now you have the development environment set up!

### Installation From PyPI (for users)

If you just want to use the `fabric-mcp` server without developing it, you can install it directly from PyPI:

```bash
# Using pip
pip install fabric-mcp

# Or using uv
uv pip install fabric-mcp
```

This will install the package and its dependencies. You can then run the server using the `fabric-mcp` command.

## Configuration (Environment Variables)

The `fabric-mcp` server can be configured using the following environment variables:

- **`FABRIC_BASE_URL`**: The base URL of the running Fabric REST API server (`fabric --serve`).
  - *Default*: `http://127.0.0.1:8080`
- **`FABRIC_API_KEY`**: The API key required to authenticate with the Fabric REST API server, if it's configured to require one.
  - *Default*: None (Authentication is not attempted if not set).
- **`FABRIC_MCP_LOG_LEVEL`**: Sets the logging verbosity for the `fabric-mcp` server itself.
  - *Options*: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (case-insensitive).
  - *Default*: `INFO`

You can set these variables in your shell environment (or put them into a `.env` file in the working directory) before running `fabric-mcp`:

```bash
export FABRIC_BASE_URL="http://your-fabric-host:port"
# This must match the key used by fabric --serve
export FABRIC_API_KEY="your_secret_api_key"
export FABRIC_MCP_LOG_LEVEL="DEBUG"

# Standard I/O transport (default)
fabric-mcp --stdio

# HTTP Streamable transport for HTTP-based MCP clients
fabric-mcp --http-streamable

# Custom host/port for HTTP transport
fabric-mcp --http-streamable --host 0.0.0.0 --port 3000 --mcp-path /message
```

### Transport Options

The `fabric-mcp` server supports multiple transport methods:

- **`--stdio`**: Standard I/O transport for direct MCP client integration (default)
- **`--http-streamable`**: HTTP-based transport that runs a full HTTP server for MCP communication
  - `--host`: Server bind address (default: 127.0.0.1)
  - `--port`: Server port (default: 8000)
  - `--mcp-path`: MCP endpoint path (default: /message)

For more details on transport configuration, see the [Infrastructure and Deployment Overview](./docs/architecture/infrastructure-and-deployment-overview.md#transport-configuration).

## Contributing

Read the [contribution document here](./docs/contributing.md) and please follow the guidelines for this repository.

Also refer to the [cheat-sheet for contributors](./docs/contributing-cheatsheet.md) which contains a micro-summary of the
development workflow.

## License

Copyright (c) 2025, [Kayvan Sylvan](kayvan@sylvan.com) Licensed under the [MIT License](./LICENSE).

[fabricGithubLink]: https://github.com/danielmiessler/fabric
[MCP]: https://modelcontextprotocol.io/
[architecture_doc]: ./docs/architecture/index.md
[develop_publish_link]: https://github.com/ksylvan/fabric-mcp/actions/workflows/publish.yml?branch=develop
[develop_publish]: https://github.com/ksylvan/fabric-mcp/actions/workflows/publish.yml/badge.svg?branch=develop
[develop_tests_link]: https://github.com/ksylvan/fabric-mcp/actions/workflows/tests.yml?branch=develop
[develop_tests]: https://github.com/ksylvan/fabric-mcp/actions/workflows/tests.yml/badge.svg?branch=develop
[mit_license_link]: https://opensource.org/licenses/MIT
[mit_license]: https://img.shields.io/badge/License-MIT-yellow.svg
[main_publish_link]: https://github.com/ksylvan/fabric-mcp/actions/workflows/publish.yml
[main_publish]: https://github.com/ksylvan/fabric-mcp/actions/workflows/publish.yml/badge.svg
[main_tests_link]: https://github.com/ksylvan/fabric-mcp/actions/workflows/tests.yml
[main_tests]: https://github.com/ksylvan/fabric-mcp/actions/workflows/tests.yml/badge.svg
[bmad-method]: https://github.com/bmadcode/BMAD-METHOD
