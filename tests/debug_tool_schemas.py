"""Dump the input schema of every tool exposed by the AClimate MCP server.

Run with:  uv run tests/debug_tool_schemas.py
"""

import asyncio
import json
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = os.getenv("ACLIMATE_MCP_URL", "https://mcp.aclimate.org/mcp")


async def main() -> None:
    print(f"Connecting to {MCP_URL}\n")

    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()

            for tool in tools.tools:
                print("=" * 70)
                print(f"TOOL: {tool.name}")

                if tool.description:
                    print(f"  desc: {tool.description.strip()[:120]}")

                print("  inputSchema:")
                print(
                    json.dumps(
                        tool.inputSchema,
                        indent=4,
                        ensure_ascii=False,
                    )
                )
                print()


if __name__ == "__main__":
    asyncio.run(main())
