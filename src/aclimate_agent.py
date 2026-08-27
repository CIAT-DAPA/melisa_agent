import json
import logging
import os
from typing import Any

from litellm import acompletion
from litellm.experimental_mcp_client import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class _RescuedFunction:
    """Mimics the .function attribute of a native tool call."""

    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _RescuedToolCall:
    """Tool call reconstructed from JSON the model emitted as plain text."""

    _counter = 0

    def __init__(self, name: str, arguments: dict) -> None:
        _RescuedToolCall._counter += 1
        self.id = f"rescued_call_{_RescuedToolCall._counter}"
        self.function = _RescuedFunction(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


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

## ALCANCE - Verifica esto ANTES del PASO 0

Solo respondes consultas sobre informacion agroclimatica: clima historico
(temperatura, precipitacion, radiacion, humedad, viento), climatologias e
indicadores agroclimaticos (dias secos consecutivos, olas de calor, etc.)
en los sitios y regiones cubiertos por AClimate (Guatemala, Honduras,
Nicaragua y Amazonia).

Si la pregunta NO es de ese tema (deportes, politica, tareas escolares,
programacion, recetas, temas personales, u otro tema general):
- NO llames ninguna tool.
- Responde cordialmente, en el idioma del usuario, con 3 partes breves:
  1) que eres un asistente especializado en informacion agroclimatica de
     AClimate y no puedes ayudar con ese tema,
  2) que si puedes hacer: consultar clima historico, climatologias e
     indicadores agroclimaticos por sitio, region o coordenadas,
  3) un ejemplo de pregunta que si puedes responder, como:
     "Cual fue la precipitacion mensual de 2023 en <un sitio>?"
- No sermonees ni te disculpes en exceso: una respuesta corta y amable.

Si es un saludo o cortesia o una despededida ("hola", "gracias", "adios"):
responde amablemente con algo similar y ofrece ayuda con informacion
agroclimatica, sin llamar tools.

Casos limite (agricultura general, cultivos, plagas, clima): explica que
tu especialidad son los datos agroclimaticos y reconduce hacia lo que si
puedes consultar. No respondas con conocimiento propio.

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

A3. Determina start_date y end_date. NUNCA llames una tool de datos sin ellas:
    - Si el usuario pidio un periodo (ej. "2020"): conviertelo a fechas
      (start_date=2020-01-01, end_date=2020-12-31) y recortalo al rango
      disponible que obtuviste en A2. Si queda totalmente fuera del rango,
      NO llames la tool: informa el rango disponible y pregunta.
    - Si el usuario NO pidio periodo: usa como start_date y end_date el rango
      completo que devolvio la tool de rangos en A2, y menciona en tu
      respuesta final que usaste todo el periodo disponible.

A4. Llama la tool de datos con TODOS sus argumentos obligatorios:
    - get_daily_climate:     location_id, start_date, end_date
    - get_monthly_climate:   location_id, start_date, end_date
    - get_indicator_history: location_id, start_date, end_date
    - get_climatology:       location_id (no lleva fechas)
    Las fechas van en formato YYYY-MM-DD y salen SIEMPRE de A3.

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
- Si una tool de datos falla por campos faltantes (ej. "start_date Field
  required"): NO repitas la llamada igual. Consulta la tool de rangos
  correspondiente (get_available_climate_*_date_ranges o
  get_available_indicator_date_ranges), toma el rango que devuelva y vuelve
  a llamar la tool de datos incluyendo start_date y end_date de ese rango.

## CIERRE

Cuando ya tengas todos los datos y hayas respondido completamente, termina respondiendo
en texto normal, sin mas tool calls."""


class AClimateAgent:
    """LLM agent that consumes tools exposed by the AClimate MCP server."""

    def __init__(
        self,
        *,
        mcp_url: str = "https://mcp.aclimate.org/mcp",
        model: str = "ollama_chat/llama3.1:8b",
        api_base: str = "http://localhost:11434",
        max_iterations: int = 15,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        num_ctx: int = 8192,
    ) -> None:
        self.mcp_url = mcp_url
        self.model = model
        self.api_base = api_base
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_ctx = num_ctx

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

        # (tool + argumentos) -> resultado ya obtenido en esta conversacion
        executed_calls: dict[str, dict[str, Any]] = {}

        # iteraciones consecutivas sin ninguna llamada nueva
        stalled_iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            logger.debug("Agent iteration %s", iteration)
            print(f"Agent iteration {iteration}")
            print(f"Memory: {self.memory}")

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
                num_ctx=self.num_ctx,
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            # llama3.1 a veces emite la tool call como texto JSON en vez de
            # usar el canal estructurado. Rescatarla antes de darla por respuesta.
            if not tool_calls and message.content:
                rescued = self._extract_text_tool_calls(message.content)

                if rescued:
                    logger.warning("Rescued %s tool call(s) emitted as plain text",len(rescued),)
                    print(f"Rescued {len(rescued)} tool call(s) emitted as plain text")
                    tool_calls = rescued
                    # No guardar el JSON crudo como contenido: el modelo
                    # tenderia a imitar ese formato en los turnos siguientes.
                    message.content = None

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

            made_progress = False

            print(f"Tool calls detected: {len(tool_calls)}")
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_arguments = self._parse_tool_arguments(
                    tool_call.function.arguments
                )

                call_key = self._build_call_key(tool_name, tool_arguments)

                if call_key in executed_calls:
                    logger.warning("Repeated call to %s with %s - serving cached result",tool_name,tool_arguments,)
                    print(f"Repeated call to {tool_name} with {tool_arguments} - serving cached result")

                    result = {
                        "repeated_call": True,
                        "note": (
                            f"Ya llamaste '{tool_name}' con estos mismos argumentos "
                            f"en esta conversacion. Abajo esta el resultado que ya "
                            f"obtuviste; no se volvio a ejecutar la tool. Usa ese dato "
                            f"para avanzar al siguiente paso del flujo, o cambia de "
                            f"estrategia (otros argumentos u otra tool). No repitas "
                            f"esta llamada."
                        ),
                        "previous_result": executed_calls[call_key],
                    }

                else:
                    logger.info("Executing MCP tool %s with arguments %s",tool_name,tool_arguments,)
                    print(f"Executing MCP tool {tool_name} with arguments {tool_arguments}")

                    result = await self._execute_tool(
                        session=session,
                        tool_name=tool_name,
                        tool_arguments=tool_arguments,
                    )

                    print(f"Result from {tool_name}: {result}")

                    executed_calls[call_key] = result
                    made_progress = True

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

            if made_progress:
                stalled_iterations = 0

            else:
                stalled_iterations += 1

                logger.warning("Iteration %s produced no new tool calls (stalled=%s)",iteration,stalled_iterations,)
                print(f"Iteration {iteration} produced no new tool calls (stalled={stalled_iterations})")

                if stalled_iterations >= 2:
                    stalled = (
                        "No fue posible avanzar: el agente repitio las mismas consultas "
                        "sin obtener informacion nueva. Por favor reformula la pregunta "
                        "o especifica la ubicacion, la variable y el periodo."
                    )

                    self.memory.append(
                        {
                            "role": "assistant",
                            "content": stalled,
                        }
                    )

                    return stalled

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
    def _extract_text_tool_calls(content: str) -> list[_RescuedToolCall]:
        """Detect tool calls that llama3.1 emitted as plain-text JSON.

        Recognized shapes (one JSON object, or several separated by newlines/';'):
          {"type": "function", "name": "tool", "parameters": {...}}
          {"name": "tool", "parameters": {...}}
          {"name": "tool", "arguments": {...}}
        Returns [] if the content is not exclusively tool-call JSON.
        """
        text = content.strip()

        # Fast reject: normal prose answers must never be treated as tool calls.
        if not text.startswith("{"):
            return []

        candidates: list[str] = []

        try:
            json.loads(text)
            candidates = [text]

        except json.JSONDecodeError:
            # Maybe several JSON objects separated by newlines or semicolons
            parts = [
                p.strip().rstrip(";")
                for p in text.replace("};", "}\n").splitlines()
            ]
            candidates = [p for p in parts if p.startswith("{")]

        rescued: list[_RescuedToolCall] = []

        for candidate in candidates:
            try:
                obj = json.loads(candidate)

            except json.JSONDecodeError:
                return []

            if not isinstance(obj, dict):
                return []

            name = obj.get("name")
            arguments = obj.get("parameters", obj.get("arguments"))

            if not isinstance(name, str) or not isinstance(arguments, dict):
                return []

            rescued.append(_RescuedToolCall(name=name, arguments=arguments))

        return rescued


    @staticmethod
    def _build_call_key(tool_name: str, tool_arguments: dict[str, Any]) -> str:
        """Identidad de una llamada: nombre + argumentos normalizados."""
        return f"{tool_name}:" + json.dumps(
            tool_arguments,
            sort_keys=True,
            default=str,
        )


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