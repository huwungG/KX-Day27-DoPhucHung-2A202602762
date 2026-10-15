"""LLM factory. Returns an OpenAI-compatible chat model.

Uses OpenAI directly when OPENAI_API_KEY is set, otherwise OpenRouter
(OPENROUTER_API_KEY). Both speak the same chat-completions API.
"""

import os

from langchain_openai import ChatOpenAI


OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    if api_key := os.environ.get("OPENAI_API_KEY"):
        base_url = os.environ.get("LLM_BASE_URL", OPENAI_BASE_URL)
        if "openrouter.ai" in base_url:
            base_url = OPENAI_BASE_URL
        # OpenRouter-style ids ("openai/gpt-4o-mini") → plain OpenAI ids
        model = model.removeprefix("openai/")
    elif api_key := os.environ.get("OPENROUTER_API_KEY"):
        base_url = os.environ.get("LLM_BASE_URL", OPENROUTER_BASE_URL)
    else:
        raise RuntimeError(
            "Set OPENAI_API_KEY or OPENROUTER_API_KEY — copy .env.example to .env"
        )

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )
