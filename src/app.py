import os

import gradio as gr
from dotenv import load_dotenv

from aclimate_agent import AClimateAgent

load_dotenv()
ACLIMATE_MCP_URL = os.getenv("ACLIMATE_MCP_URL", "https://mcp.aclimate.org/mcp")
ACLIMATE_AGENT_MODEL = os.getenv("ACLIMATE_AGENT_MODEL", "ollama_chat/llama3.1:8b")
ACLIMATE_AGENT_API_BASE = os.getenv("ACLIMATE_AGENT_API_BASE", "http://localhost:11434")
PORT = int(os.getenv("ACLIMATE_AGENT_PORT", 7860))

# Cuantos mensajes recientes del historial se le pasan al modelo.
# Controla el uso de la ventana de contexto (num_ctx) en conversaciones largas.
MAX_HISTORY_MESSAGES = 20


def build_memory_from_history(history: list[dict]) -> list[dict]:
    """Rebuild the agent memory from Gradio's per-session chat history.

    Gradio keeps one history per browser session and passes it on every
    call, so the agent itself can stay stateless: no memory is shared
    between users, and the conversation survives across turns.
    """
    memory = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
        if msg.get("role") in ("user", "assistant")
        and isinstance(msg.get("content"), str)
        and msg["content"].strip()
    ]
    return memory[-MAX_HISTORY_MESSAGES:]


async def chat(message, history):
    # Un agente nuevo por llamada: sin estado compartido entre usuarios.
    # La sesion vive en el history que Gradio mantiene por navegador.
    agent = AClimateAgent(
        mcp_url=ACLIMATE_MCP_URL,
        model=ACLIMATE_AGENT_MODEL,
        api_base=ACLIMATE_AGENT_API_BASE,
    )
    agent.memory = build_memory_from_history(history)

    return await agent.chat(message)


app = gr.ChatInterface(
    fn=chat,
    title="Melisa - AClimate Agent",
    description="AI assistant for agroclimatic information available for Guatemala, Honduras, Nicaragua, and Amazonía",
)

app.launch(server_port=PORT)
