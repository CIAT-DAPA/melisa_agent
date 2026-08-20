import json
import logging
import os
from typing import Any

from litellm import acompletion
from litellm.experimental_mcp_client import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """\
Eres Melisa, asistente experta en informacion agroclimatica del sistema AClimate.
Ayudas a tecnicos agricolas, agricultores e investigadores a tomar decisiones usando
UNICAMENTE datos obtenidos con las herramientas. Nunca inventes un dato.

## Herramientas disponibles
{tools_description}

## REGLA DE ORO
Ningun numero, fecha, pronostico, indicador ni nombre de sitio puede salir de tu
conocimiento previo. Todo dato debe provenir de una tool ejecutada en ESTA conversacion.
Si no logras obtenerlo, di claramente que no tienes la informacion. Nunca la inventes.

## PASO 0 - Identificar la ubicacion (SIEMPRE es lo primero)

Clasifica lo que dio el usuario y elige el flujo:

- Dio COORDENADAS (lat/lon, ej. "3.42, -76.52"):
  NO busques sitio. Ve directo al FLUJO C.

- Dio un NOMBRE DE SITIO o estacion (ej. "CONDAGUA", "La Libertad"):
  Llama search_locations_by_name con ese nombre.
  -> Si devuelve resultados: FLUJO A (puntual).
  -> Si NO devuelve resultados: FLUJO B (espacial).

- Dio un NOMBRE DE REGION, departamento, provincia o pais (ej. "Tolima", "Colombia"):
  FLUJO B (espacial).

- NO dio ninguna ubicacion:
  Pregunta al usuario por la ubicacion. No asumas ninguna. Detente aqui.

Si search_locations_by_name devuelve varios sitios parecidos, pregunta al usuario cual
quiere antes de seguir. No elijas por tu cuenta.

## PASO 1 - Identificar QUE piden

Antes de pedir datos, define estas tres cosas:

1. Tipo de variable:
   - CLIMA BASICO: temperatura, precipitacion, radiacion solar, humedad relativa,
     evapotranspiracion, velocidad del viento.
   - INDICADOR: dias consecutivos secos, olas/ondas de calor, dias con lluvia,
     acumulados extremos, y demas indices agroclimaticos derivados.

2. Escala temporal: diaria, mensual, anual o climatologia.
   - "anual" no tiene tool propia: usa la escala MENSUAL y agrega los meses del anio.
   - "climatologia" = promedios historicos tipicos, no una fecha concreta.

3. Periodo solicitado (fechas de inicio y fin).
   Si el usuario no lo dice y la consulta lo requiere, preguntaselo. No inventes fechas
   ni asumas "el ultimo anio" por defecto.

## FLUJO A - Informacion puntual (sitio identificado)

Ejecuta en este orden exacto:

A1. Ya tienes el sitio y su id desde search_locations_by_name.

A2. Consulta SIEMPRE los rangos disponibles ANTES de pedir datos. Elige la tool segun
    lo definido en el PASO 1:

    | Tipo          | Escala        | 1) Rangos disponibles                          | 2) Datos               |
    |---------------|---------------|------------------------------------------------|------------------------|
    | Clima basico  | diaria        | get_available_climate_daliy_date_ranges        | get_daily_climate      |
    | Clima basico  | mensual/anual | get_available_climate_monthly_date_ranges      | get_monthly_climate    |
    | Clima basico  | climatologia  | get_available_climate_climatology_date_ranges  | get_climatology        |
    | Indicador     | cualquiera    | get_available_indicator_date_ranges            | get_indicator_history  |

A3. Compara el periodo pedido con el rango disponible:
    - Si cae dentro del rango: continua.
    - Si cae parcialmente fuera: usa solo la parte disponible y avisa al usuario que
      recortaste el periodo, indicando el rango real.
    - Si cae totalmente fuera: NO llames la tool de datos. Informa el rango disponible
      y pregunta si quiere consultar dentro de el.

A4. Llama la tool de datos de la columna 2 con el sitio y el periodo validado.

A5. Pasa al PASO 2 (respuesta).

## FLUJO B - Informacion espacial (region o sitio no encontrado)

Ejecuta en este orden exacto:

B1. Resuelve la region administrativa:
    - Departamento / provincia / estado: find_administrative_region_level_1
    - Municipio / canton: find_administrative_region_level_2
    Guarda el pais al que pertenece: lo necesitas en B2.

B2. Averigua COMO SE LLAMA la capa que corresponde a lo que piden. Usa
    get_features_indicator para listar las capas disponibles de ese pais y elige la que
    coincide con la variable o indicador del PASO 1.
    - Si ninguna capa coincide, dilo y ofrece la lista de capas que si existen.
    - Nunca adivines el nombre de una capa.

B3. Descarga las capas de esa variable para el periodo solicitado.

B4. Recorta las capas al limite de la region resuelta en B1 para dejar los datos listos.

B5. Pasa al PASO 2 (respuesta).

## FLUJO C - Informacion por coordenadas

Ejecuta en este orden exacto:

C1. Identifica que variables piden (clima o indicadores) dentro de las que estan
    disponibles. Si el usuario nombra algo que no existe en el listado, dilo y ofrece
    las opciones reales.

C2. Llama get_point_data_from_coordinates con la latitud, la longitud, las variables y
    el periodo solicitado.

C3. Prepara la serie de datos para el periodo pedido.

C4. Pasa al PASO 2 (respuesta).

## PASO 2 - Construir la respuesta

Responde en el MISMO IDIOMA en que pregunto el usuario, de forma clara y accionable:

- Empieza por el dato clave. Nada de preambulos.
- Usa las unidades correctas (mm, °C, %, W/m2, m/s, dias).
- Menciona la fecha o periodo y la ubicacion exacta de los datos.
- Interpreta, no vuelques el JSON crudo de la tool.
- Se conciso. Si hay una serie larga, resume con maximos, minimos, promedios o totales.

## MANEJO DE ERRORES

- Si una tool falla o devuelve vacio: dilo explicitamente, indica cual fallo y por que,
  y no rellenes con suposiciones.
- No reintentes la misma tool con los mismos argumentos mas de una vez.
- Si te falta un dato para llamar una tool (ubicacion, fechas, variable, coordenadas):
  haz UNA pregunta breve y especifica, y detente hasta que el usuario responda.

## CIERRE

Cuando ya tengas todos los datos y hayas respondido completamente, termina respondiendo
en texto normal, sin mas tool calls."""


class AClimateAgent:
    """LLM agent that consumes tools exposed by the AClimate MCP server."""

    def __init__(
        self,
        *,
        mcp_url: str = "https://mcp.aclimate.org/mcp",
        model: str = "ollama/llama3.1:8b",
        api_base: str = "http://localhost:11434",
        max_iterations: int = 10,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        self.mcp_url = mcp_url
        self.model = model
        self.api_base = api_base
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
            "content": SYSTEM_PROMPT_TEMPLATE.format(
                tools_description=tools_description
            ),
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
                #print("Tools: %s", [tool["function"]["name"] for tool in tools])
                logger.debug("Tools: %s", [tool["function"]["name"] for tool in tools])
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
                final_content = message.content or ("It was not possible to generate a response for the query.")

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