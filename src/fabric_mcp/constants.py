"""Constants used throughout the fabric-mcp package."""

DEFAULT_MCP_HTTP_PATH = "/message"

# Sensitive configuration key patterns for redaction
SENSITIVE_CONFIG_PATTERNS = ["*_API_KEY", "*_TOKEN", "*_SECRET", "*_PASSWORD"]

# API key prefixes that indicate a value is an API key regardless of key name
API_KEY_PREFIXES = ["sk-", "ant-", "xai-", "gsk_", "AIza"]
