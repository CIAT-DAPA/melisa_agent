import asyncio
import os

from aclimate_agent import AClimateAgent
from dotenv import load_dotenv

load_dotenv()
ACLIMATE_MCP_URL = os.getenv("ACLIMATE_MCP_URL", "https://mcp.aclimate.org/mcp")
ACLIMATE_AGENT_MODEL = os.getenv("ACLIMATE_AGENT_MODEL", "ollama/llama3.1:8b")
ACLIMATE_AGENT_API_BASE = os.getenv("ACLIMATE_AGENT_API_BASE", "http://192.168.199.91:11434")

async def main():
    agent = AClimateAgent(mcp_url=ACLIMATE_MCP_URL, model=ACLIMATE_AGENT_MODEL, api_base=ACLIMATE_AGENT_API_BASE)

    response = await agent.chat(
        "Está CONDAGUA en los sitios disponibles?"
    )

    print(response)


if __name__ == "__main__":
    asyncio.run(main())