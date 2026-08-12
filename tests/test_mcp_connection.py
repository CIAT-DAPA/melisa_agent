import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import (streamablehttp_client,)


MCP_URL = "https://mcp.aclimate.org/mcp"


async def main() -> None:

    print(f"Connecting to {MCP_URL}")

    async with streamablehttp_client(MCP_URL) as (read,write,_,):
        async with ClientSession(read,write,) as session:

            result = await session.initialize()
            print("MCP initialized:",result,)

            tools = await session.list_tools()
            print("\nAvailable tools:")

            for tool in tools.tools:
                print("-",tool.name,)


if __name__ == "__main__":
    asyncio.run(main())