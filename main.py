import openai
from typing import List, TypeAlias, Optional, Dict, Any
from dataclasses import dataclass
import os
import json
import subprocess
import argparse
from datetime import datetime
from openai.types.chat import ChatCompletionMessageToolCall


@dataclass
class LLMMessage:
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = (
        None  # used by the llm if it calls tools
    )
    tool_call_id: Optional[str] = None  # used for responses by a called tool


LLM_Conversation: TypeAlias = List[LLMMessage]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY
openai_model = "gpt-4o-mini"


def _print_conversation_history(conversation_history: LLM_Conversation):
    print(f"{'=' * 120}")
    for msg in conversation_history:
        print(
            f"Role: {msg.role}\n"
            f"Content: {msg.content}\n"
            f"Tool Calls: {msg.tool_calls}\n"
            f"Tool Call Id: {msg.tool_call_id}\n"
            f"{'=' * 40}"
        )


def call_openai_api(
    system_prompt: str, query: str = None, messages: LLM_Conversation = None
) -> LLM_Conversation:
    """
    Call the OpenAI API with the given query and conversation history.

    Args:
        query (str): The input query for the API.
        conversation_history (LLM_Conversation, optional): Previous messages in the conversation.

    Returns:
        LLM_Conversation: The updated conversation history.
    """
    print(f"Calling OpenAI API with query: {query}")

    if messages is None or not messages:
        messages = [LLMMessage(role="system", content=system_prompt)]

    if query is not None:
        messages: LLM_Conversation = [
            *messages,
            LLMMessage(role="user", content=query),
        ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "call_intern",
                "description": "Call an llm intern function.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to call the llm intern function with.",
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "System prompt for the intern LLM.",
                        },
                    },
                    "required": ["query", "system_prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": """Execute Python code and return the result.

                IMPORTANT!:
                Remember to always use print() in your code to return the result.
                If you don't do this, the result will not be returned.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute.",
                        },
                    },
                    "required": ["code"],
                },
            },
        },
    ]

    try:
        response = openai.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": msg.role,
                    "content": msg.content
                    if msg.content is not None
                    else "",  # Replace null with empty string (null response happens in the case of tool calls)
                    "tool_calls": msg.tool_calls,
                    "tool_call_id": msg.tool_call_id,
                }
                for msg in messages
            ],
            tools=tools,
            tool_choice="auto",
        )

        # Extract the assistant's response
        assistant_message = LLMMessage(
            role="assistant",
            content=response.choices[0].message.content,
            tool_calls=response.choices[0].message.tool_calls,
        )

        # append the assistant's response to the conversation history
        messages.append(assistant_message)

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                if (
                    tool_call.type == "function"
                    and tool_call.function.name == "call_intern"
                ):
                    tool_call_id = tool_call.id
                    tool_call_arguments = json.loads(tool_call.function.arguments)

                    query = tool_call_arguments["query"]
                    system_prompt = tool_call_arguments["system_prompt"]

                    tool_response = call_intern(query, system_prompt)

                    messages.append(
                        LLMMessage(
                            role="tool",
                            content=json.dumps(
                                tool_response
                            ),  # tool responses are returned to the llm encoded in json
                            tool_call_id=tool_call_id,
                        )
                    )

                elif tool_call.function.name == "execute_python_code":
                    tool_call_id = tool_call.id
                    tool_call_arguments = json.loads(tool_call.function.arguments)
                    code = tool_call_arguments["code"]
                    tool_response = execute_python_code(code)
                    messages.append(
                        LLMMessage(
                            role="tool",
                            content=json.dumps(tool_response),
                            tool_call_id=tool_call_id,
                        )
                    )

                else:
                    # if tool call is not of type function, raise an error
                    raise ValueError(f"Unknown tool call: {tool_call}")

            # we collect all tool responses and return them as a single response
            messages = call_openai_api(
                system_prompt,
                messages=messages,
            )

        return messages

    except Exception as e:
        print(f"An error occurred: {e}")
        return messages


def call_intern(query: str, system_prompt: str) -> str:
    """
    Call the OpenAI API with the given query and system prompt.
    """
    messages = call_openai_api(system_prompt, query)

    # extract the assistant's response
    response_content = messages[-1].content

    return response_content


def execute_python_code(code: str) -> str:
    """
    Execute Python code and return the result.
    """
    print(f"Executing Python code: {code}")

    # Specify the path to the desired Python interpreter
    python_executable = "/opt/homebrew/bin/python3.11"

    # Create a temporary file to store the code
    with open("temp_code.py", "w") as f:
        f.write(code)

    try:
        # Run the code using the specified Python interpreter
        result = subprocess.run(
            [python_executable, "temp_code.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"
    finally:
        # Clean up the temporary file
        os.remove("temp_code.py")

    print(f"Python code output: {output}")
    return output.strip()


def write_conversation_to_json(conversation_history: LLM_Conversation, query: str):
    def serialize_tool_call(tool_call):
        if isinstance(tool_call, ChatCompletionMessageToolCall):
            return {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        return tool_call

    output = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "conversation": [
            {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": [serialize_tool_call(tc) for tc in msg.tool_calls]
                if msg.tool_calls
                else None,
                "tool_call_id": msg.tool_call_id,
            }
            for msg in conversation_history
        ],
    }

    # Create a 'conversations' folder if it doesn't exist
    conversations_folder = "conversations"
    os.makedirs(conversations_folder, exist_ok=True)

    # Generate filename with timestamp
    filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Full path for the file
    file_path = os.path.join(conversations_folder, filename)

    with open(file_path, "w") as f:
        json.dump(output, f, indent=2)

    return file_path


def main():
    parser = argparse.ArgumentParser(description="Query the OpenAI API.")
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="The query to ask the OpenAI API.",
    )
    args = parser.parse_args()

    system_prompt = """You are a helpful assistant.
    You are a very intelligent engineering manager.
    You should plan how to best solve a problem and then delegate sub-tasks to your interns which can be called with the call_intern function.e
    Your interns are very intelligent. Your job is to take their work and combine it into a cohesive response.
    Always reject work that is not up to standard and ask the intern to improve it.

    You can also call the execute_python_code function to execute python code. This is useful for tasks that require code execution or mathematical calculations.
    IMPORTANT!:
    Remember to always use print() in your code to return the result.
    If you don't do this, the result will not be returned.
    """

    conversation_history = call_openai_api(system_prompt, args.query)

    json_filename = write_conversation_to_json(conversation_history, args.query)
    print(f"Conversation written to {json_filename}")

    _print_conversation_history(conversation_history)


if __name__ == "__main__":
    main()
