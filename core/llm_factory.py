"""
LLM factory – returns a LangChain chat model based on settings.
Always reads env vars at call time so sidebar key changes take effect immediately.
"""
import os


def get_llm(temperature: float = 0.3):
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model    = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GEMINI_API_KEY", ""),
            temperature=temperature,
            max_output_tokens=4096,
            convert_system_message_to_human=True,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            temperature=temperature,
            max_tokens=4096,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        temperature=temperature,
        max_tokens=4096,
    )
