from openai import (
    OpenAI,
)  # It is confusing, but they acttually use the same API structure as OpenAI
import os
import requests
from typing import List, TypedDict


def get_perplexity_results(query: str) -> str:
    """
    Get results from Perplexity.ai
    """
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY is not set")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant who is given a query and is supposed to generate a set of results that can be used to answer the query."
            ),
        },
        {
            "role": "user",
            "content": (query),
        },
    ]

    client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

    # chat completion without streaming
    response = client.chat.completions.create(
        model="llama-3-sonar-small-32k-online",
        messages=messages,
    )

    return response.choices[0].message.content


class BingResult(TypedDict):
    name: str
    url: str
    snippet: str


def get_bing_search_results(query: str, count: int = 5) -> List[BingResult]:
    """
    Get results from Bing Search API
    """
    BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")

    if not BING_SEARCH_API_KEY:
        raise ValueError("BING_SEARCH_API_KEY is not set")

    headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_API_KEY}

    params = {"q": query, "count": count, "responseFilter": "Webpages", "mkt": "en-US"}

    response = requests.get(
        "https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params
    )

    response.raise_for_status()
    search_results = response.json()

    return [
        BingResult(name=result["name"], url=result["url"], snippet=result["snippet"])
        for result in search_results.get("webPages", {}).get("value", [])
    ]
