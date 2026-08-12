import json
import logging
import os
from typing import Any

from litellm import acompletion
from litellm.experimental_mcp_client import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class AClimateAgent:
    """LLM agent that consumes tools exposed by the AClimate MCP server."""

    def __init__(
        self,
        *,
        mcp_url: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        max_iterations: int = 10,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        self.mcp_url = mcp_url or os.getenv(
            "ACLIMATE_MCP_URL",
            "https://mcp.aclimate.org/mcp",
        )
        self.model = model or os.getenv(
            "ACLIMATE_AGENT_MODEL",
            "ollama/llama3.1:8b",
        )
        self.api_base = api_base or os.getenv(
            "ACLIMATE_AGENT_API_BASE",
            "http://192.168.199.91:11434",
        )

        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.memory: list[dict[str, Any]] = []


    @staticmethod
    def build_system_prompt(tools: list[dict[str, Any]]) -> dict[str, str]:
        tools_description = "\n".join(
            (
                f"- {tool['function']['name']}: "
                f"{tool['function'].get('description', 'Sin descripcion')}"
            )
            for tool in tools
        )

        return {
            "role": "system",
            "content": f"""
            Eres un asistente experto en informacion agroclimatica, parte del sistema AClimate.
            Ayudas a técnicos agricolas, agricultores e investigadores a tomar decisiones informadas usando datos
            reales obtenidos mediante las herramientas disponibles. Nunca inventes datos.

            ## Herramientas disponibles
            {tools_description}

            1. **Nunca inventes datos climáticos, pronósticos, ni recomendaciones agronómicas.**
            Si la pregunta requiere datos (clima actual, pronóstico, históricos, índices
            agroclimáticos, etc.), SIEMPRE usa la tool correspondiente antes de responder.

            2. **Si falta información para usar una tool** (ubicación, cultivo, rango de fechas,
            coordenadas, nombre de estación, etc.), pregunta al usuario de forma breve y
            específica antes de llamar la tool. No asumas ubicaciones ni fechas por defecto.

            3. **Si una tool falla o no devuelve datos**, informa claramente al usuario que no
            fue posible obtener la información y por qué, en vez de rellenar con suposiciones.

            4. **Al presentar resultados**, hazlo en el mismo idioma en que se hizo la pregunta, de forma clara y accionable:
            - Resume el dato clave primero.
            - Usa unidades correctas (mm, °C, %, etc.).
            - Si es relevante, menciona la fecha/periodo y la ubicación de los datos.

            5. **Sé conciso.** No repitas literalmente los datos crudos de la tool; interpreta y
            resume lo relevante para la pregunta del usuario.

            Cuando ya tengas toda la información necesaria y hayas respondido completamente al
            usuario, termina la conversación respondiendo normalmente sin más tool calls.
            """.strip(),
        }


    def reset_memory(self) -> None:
        self.memory.clear()


    async def chat(self, user_message: str) -> str:
        """Process a user message through the LLM and AClimate MCP tools."""

        if not user_message.strip():
            return "Please provide a non-empty message to process."

        self.memory.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # Keep the MCP context open for the complete tool-calling loop
        async with streamablehttp_client(self.mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Load tools using the current MCP session
                tools = await load_mcp_tools(
                    session=session,
                    format="openai",
                )

                system_prompt = self.build_system_prompt(tools)

                return await self._run_agent_loop(
                    session=session,
                    tools=tools,
                    system_prompt=system_prompt,
                )


    async def _run_agent_loop(
        self,
        session: ClientSession,
        tools: list[dict[str, Any]],
        system_prompt: dict[str, str],
    ) -> str:

        for iteration in range(1, self.max_iterations + 1):
            logger.debug("Agent iteration %s", iteration)

            response = await acompletion(
                model=self.model,
                api_base=self.api_base,
                messages=[
                    system_prompt,
                    *self.memory,
                ],
                tools=tools,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                final_content = message.content or (
                    "It was not possible to generate a response for the query."
                )

                self.memory.append(
                    {
                        "role": "assistant",
                        "content": final_content,
                    }
                )

                return final_content

            self.memory.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        tool_call.model_dump()
                        for tool_call in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_arguments = self._parse_tool_arguments(
                    tool_call.function.arguments
                )

                logger.info(
                    "Executing MCP tool %s with arguments %s",
                    tool_name,
                    tool_arguments,
                )

                result = await self._execute_tool(
                    session=session,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                )

                self.memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(
                            result,
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )

        fallback = (
            "It was not possible to complete the request within the maximum number of allowed iterations."
        )

        self.memory.append(
            {
                "role": "assistant",
                "content": fallback,
            }
        )

        return fallback


    async def _execute_tool(
        self,
        session: ClientSession,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> dict[str, Any]:

        try:
            mcp_result = await session.call_tool(
                name=tool_name,
                arguments=tool_arguments,
            )

            text_blocks = [
                block.text
                for block in mcp_result.content
                if hasattr(block, "text")
            ]

            result: dict[str, Any] = {
                "content": text_blocks,
                "is_error": bool(
                    getattr(mcp_result,"isError",False,)
                ),
            }

            structured_content = getattr(
                mcp_result,
                "structuredContent",
                None,
            )

            if structured_content is not None:
                result["structured_content"] = structured_content

            return result

        except Exception as exc:
            logger.exception("Error executing MCP tool %s",tool_name,)

            return {
                "error": f"Error executing {tool_name}: {exc}",
                "is_error": True,
            }


    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments

        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)

        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid JSON arguments received from the model: {arguments}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Tool arguments must decode to a JSON object."
            )

        return parsed