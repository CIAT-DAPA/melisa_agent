import asyncio
import gradio as gr

from aclimate_agent import AClimateAgent

agent = AClimateAgent()

async def chat(message, history):
    return await agent.chat(message)

demo = gr.ChatInterface(
    fn=chat,
    title="AClimateAI",
    description="AI assistant for agroclimatic information",
)

demo.launch()