# Changelog

## [Unreleased]

### Added
- Implemented real pattern listing: `fabric_list_patterns` now fetches the true list of patterns from the Fabric backend (`/patterns/names`), replacing the placeholder.
- Implemented real pattern prompt retrieval: `fabric_pattern_prompt` (renamed from `fabric_pattern_details`) now fetches the full prompt for a given pattern from the Fabric backend (`/patterns/{pattern_name}`) and returns it, with no placeholder values.
- Cleaned up the MCP tool interface: removed the unused description field from pattern prompt responses.
- Updated documentation: Added `llm.md` with LLM- and developer-friendly usage instructions for Fabric and the MCP server.

### Changed
- Renamed `fabric_pattern_details` to `fabric_pattern_prompt` for clarity and accuracy.

### Fixed
- Eliminated all placeholder responses for core MCP tools (list patterns, get pattern prompt).

---
All changes are staged for PR to https://github.com/Gaemghost20000/fabric-mcp.
