# ibex_agent.py
import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from agente.ibex_utils import get_status, script_full_load, script_incremental

# ejecutar python -m agente.ibex_agent

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Devuelve si cada ticker existe ya en la BBDD y la última fecha registrada.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["tickers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "script_full_load",
            "description": "Realiza una carga histórica completa para los tickers indicados.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "from_date": {"type": "string"}
                },
                "required": ["tickers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "script_incremental",
            "description": "Actualiza los tickers indicados desde su última fecha hasta hoy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["tickers"]
            }
        }
    },
]


TOOLS_MAP = {
    "get_status": get_status,
    "script_full_load": script_full_load,
    "script_incremental": script_incremental,
}


SYSTEM_PROMPT = """
Eres un agente IBEX encargado de mantener actualizada la tabla stock_market
de una base de datos SQLite.

Tu procedimiento debe ser:

1. Siempre que el usuario te pase tickers, primero llama a get_status
para saber si existen en la BBDD y cuál es su última fecha.

2. Con esa información:
- Para los tickers que NO existan (exists = false):
    usa script_full_load para hacer una carga histórica completa
    desde 2023-01-01 hasta la fecha actual.
- Para los tickers que SÍ existan:
    usa script_incremental para actualizarlos desde su última fecha
    hasta hoy.

Toma estas decisiones tú mismo, sin pedir confirmación al usuario.

3. No inventes datos ni digas que has hecho algo que no has hecho.
Solo describe el resultado de las herramientas que hayas llamado.

4. Tu respuesta final debe ser un resumen en español, claro y breve,
indicando para cada ticker:
    - si ya existía o no,
    - si has hecho carga completa o actualización incremental,
    - cuántas filas nuevas se han insertado
    - y cuál es la última fecha registrada.
"""

def run_ibex_agent(tickers: List[str]) -> str:
    """
    Ejecuta el agente con tool calling y devuelve un resumen natural-language
    de todo el proceso, usando las banderas que devuelven las tools.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Comprueba y actualiza estos tickers: {tickers}."}
    ]

    tool_results: Dict[str, Dict[str, Any]] = {}  # <-- almacenamos resultados

    while True:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # ¿Quiere llamar tools?
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            })

            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                print(f"[AGENTE] Llama a {name} con args={args}")
                fn = TOOLS_MAP[name]
                result = fn(**args)
                print(f"[AGENTE] Resultado {name}: {result}")

                # guardamos resultados por si necesitamos un resumen final
                if isinstance(result, dict):
                    for k, v in result.items():
                        tool_results[k] = v

                # devolvemos el resultado al modelo
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": json.dumps(result, default=str),
                })
            continue

        # ------------------------------------------------------------------
        # Resumen natural-language basado en tool_results
        # ------------------------------------------------------------------
        resumen = "\nResumen final del proceso:\n\n"

        for ticker in tickers:
            T = ticker.upper()

            if T not in tool_results:
                resumen += f"- {T}: No se obtuvo información.\n"
                continue

            info = tool_results[T]
            status = info.get("status", "")

            if status == "full_load":
                resumen += (
                    f"- {T}: Se realizó una carga completa desde {info['from']} "
                    f"hasta {info['to']}, insertando {info['new_rows']} filas.\n"
                )
                if info.get("used_fallback"):
                    resumen += (
                        f"   • Se detectó que YF no llegaba a la fecha actual y se "
                        f"rellenó el tramo final con {info['fallback_rows']} filas desde EODHD.\n"
                    )

            elif status == "updated":
                resumen += (
                    f"- {T}: Se actualizó de forma incremental, insertando "
                    f"{info['new_rows']} nuevas filas. Última fecha registrada: {info['last_date']}.\n"
                )
                if info.get("used_fallback"):
                    resumen += (
                        f"   • No había datos recientes en YF y se añadieron "
                        f"{info['fallback_rows']} filas finales desde EODHD.\n"
                    )

            elif status == "no_new_data":
                resumen += (
                    f"- {T}: Ya estaba actualizado. Última fecha: {info['last_date']}.\n"
                )

            elif status == "no_data_in_db":
                resumen += (
                    f"- {T}: No existía en la base de datos y no pudo hacerse incremental.\n"
                )

            elif status == "too_old_for_incremental":
                resumen += (
                    f"- {T}: La última fecha ({info['last_date']}) está demasiado desfasada "
                    f"({info['gap_days']} días). El incremento no es posible con la cuenta free. "
                    f"Debe realizarse un full load.\n"
                )

            else:
                resumen += f"- {T}: Estado desconocido.\n"

        return resumen
    
