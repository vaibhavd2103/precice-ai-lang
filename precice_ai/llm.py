from langchain_openai import ChatOpenAI

from precice_ai import config


def build_chat_model() -> ChatOpenAI:
    if not config.llm_is_configured():
        raise ValueError(
            "LLM is not configured. Set PRECICE_AI_API_KEY or LLM_API_KEY (or OPENROUTER_API_KEY), "
            "plus a model if you do not want the default."
        )

    kwargs = {
        "model": config.MODEL,
        "openai_api_key": config.LLM_API_KEY,
        "streaming": True,
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    if config.LLM_BASE_URL:
        kwargs["openai_api_base"] = config.LLM_BASE_URL

    if config.LLM_PROVIDER == "openrouter":
        kwargs["default_headers"] = {
            "HTTP-Referer": "https://precice.org",
            "X-Title": "preCICE AI",
        }

    return ChatOpenAI(**kwargs)
