"""
LLM factory – returns a LangChain chat model based on settings.
Supports: gemini (default), openai, anthropic
"""
from config.settings import LLM_PROVIDER, LLM_MODEL, OPENAI_KEY, ANTHROPIC_KEY, GEMINI_KEY


def get_llm(temperature: float = 0.3):
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_KEY,
            temperature=temperature,
            max_output_tokens=4096,
            convert_system_message_to_human=True,  # Gemini requires this
        )
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL,
            anthropic_api_key=ANTHROPIC_KEY,
            temperature=temperature,
            max_tokens=4096,
        )
    # fallback: openai
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_KEY,
        temperature=temperature,
        max_tokens=4096,
    )
