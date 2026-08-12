# Melisa Agent

AI assistant for agroclimatic information powered by the [AClimate](https://aclimate.org) MCP server. Provides a Gradio chat interface where agricultural technicians, farmers, and researchers can query real climate data through natural language.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (package manager)
- Python 3.10
- An LLM backend accessible via [LiteLLM](https://docs.litellm.ai/) (default: Ollama with `llama3.1:8b`)

## Setup

```bash
# Clone the repository
git clone 
cd melisa_agent

# Install dependencies
uv sync
```

## Configuration

The agent is configured via environment variables. Create a `.env` file or export them in your shell:

| Variable | Default | Description |
|---|---|---|
| `ACLIMATE_MCP_URL` | `https://mcp.aclimate.org/mcp` | AClimate MCP server URL |
| `ACLIMATE_AGENT_MODEL` | `ollama/llama3.1:8b` | LiteLLM model identifier |
| `ACLIMATE_AGENT_API_BASE` | `http://localhost:11434` | LLM API base URL (e.g. your Ollama instance) |

Example using a different Ollama host:

```bash
export ACLIMATE_AGENT_MODEL="ollama/llama3.1:8b"
export ACLIMATE_AGENT_API_BASE="http://localhost:11434"
```

## Running the app

```bash
# Linux
uv run src/app.py

# Windows
uv run src\app.py
```

The Gradio interface will be available at `http://localhost:7860`.

## Running the tests

Both test scripts are standalone (no pytest required) and are run directly with `uv run`.

**Test 1 — MCP connection:** verifies connectivity to the AClimate MCP server and lists available tools.

```bash
# Linux
uv run tests/test_mcp_connection.py

# Windows
uv run tests\test_mcp_connection.py
```

**Test 2 — Agent integration:** sends a sample query through the full agent loop and prints the response.

```bash
# Linux
uv run tests/test_agent.py

# Windows
uv run tests\test_agent.py
```

## Project structure

```
melisa_agent/
├── src/
│   ├── app.py              # Gradio chat interface (entry point)
│   └── aclimate_agent.py   # AClimateAgent class
├── tests/
│   ├── test_agent.py       # End-to-end agent test
│   └── test_mcp_connection.py  # MCP connectivity test
├── pyproject.toml
├── .python-version         # Python 3.10
└── uv.lock
```
