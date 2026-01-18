import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from env_loader import OPENAI_API_KEY
from tracing_utils import trace_agent


client = OpenAI(api_key=OPENAI_API_KEY)   # ✅ Correct variable name


def run_llm(
    prompt: str,
    tools: Optional[List[Dict]] = None,
    tool_functions: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Run an LLM request that optionally supports tools.
    
    Args:
        prompt (str): The system or user prompt to send.
        tools (list[dict], optional): Tool schema list for model function calling.
        tool_functions (dict[str, callable], optional): Mapping of tool names to Python functions.
        model (str): Model name to use (default: gpt-4o-mini).

    Returns:
        str: Final LLM response text.
    """

    # Step 1: Initial LLM call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        tools=tools if tools else None,
        tool_choice="auto" if tools else None
    )

    message = response.choices[0].message
    print("Initial LLM Response:", message)

    # Step 2: If no tools or no tool calls, return simple model response
    if not getattr(message, "tool_calls", None):
        return message.content

    # Step 3: Handle tool calls dynamically
    if not tool_functions:
        return message.content + "\n\n⚠️ No tool functions provided to execute tool calls."

    tool_messages = []
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")
        tool_fn = tool_functions.get(func_name)

        try:
            result = tool_fn(**args) if tool_fn else {"error": f"Tool '{func_name}' not implemented."}
        except Exception as e:
            result = {"error": str(e)}

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # Step 4: Second pass — send tool outputs back to the model
    followup_messages = [
        {"role": "system", "content": prompt},
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                } for tc in message.tool_calls
            ],
        },
        *tool_messages,
    ]

    final = client.chat.completions.create(model=model, messages=followup_messages)
    return final.choices[0].message.content