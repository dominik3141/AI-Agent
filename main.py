import openai
from typing import List, TypeAlias, Optional, Dict, Any
from dataclasses import dataclass
import os
import json


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
                    },
                },
            },
        }
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

                    tool_response = call_intern(query)

                    messages.append(
                        LLMMessage(
                            role="tool",
                            content=json.dumps(
                                tool_response
                            ),  # tool responses are returned to the llm encoded in json
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


def call_intern(query: str) -> str:
    """
    Call the OpenAI API with the given query.
    """

    system_prompt = "You are a helpful assistant. Do not use any tools."

    messages = call_openai_api(system_prompt, query)

    # extract the assistant's response
    response_content = messages[-1].content

    return response_content


def main():
    system_prompt = """You are a helpful assistant.
    You can call the call_intern function to call an llm intern function.
    If you call the call_intern function. You should always call it twice with slightly different queries. This way you can get a more accurate answer.
    """
    # system_prompt = "You are a helpful assistant. Never call any tools."

    message = "How old is the earth?"
    conversation_history = call_openai_api(system_prompt, message)

    _print_conversation_history(conversation_history)


if __name__ == "__main__":
    main()
