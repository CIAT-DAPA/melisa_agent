import asyncio
import gradio as gr
import os

from dotenv import load_dotenv

from aclimate_agent import AClimateAgent

load_dotenv()
ACLIMATE_MCP_URL = os.getenv("ACLIMATE_MCP_URL", "https://mcp.aclimate.org/mcp")
ACLIMATE_AGENT_MODEL = os.getenv("ACLIMATE_AGENT_MODEL", "ollama/llama3.1:8b")
ACLIMATE_AGENT_API_BASE = os.getenv("ACLIMATE_AGENT_API_BASE", "http://localhost:11434")
PORT = int(os.getenv("ACLIMATE_AGENT_PORT", 7860))

agent = AClimateAgent(mcp_url=ACLIMATE_MCP_URL, model=ACLIMATE_AGENT_MODEL, api_base=ACLIMATE_AGENT_API_BASE)

async def chat(message, history):
    return await agent.chat(message)

app = gr.ChatInterface(
    fn=chat,
    title="Melisa - AClimate Agent",
    description="AI assistant for agroclimatic information",
)

app.launch(server_port=PORT)