"""
LLM factory – returns a LangChain chat model based on settings.
Always reads env vars at call time so changes to os.environ take effect immediately.

Bug fixes applied:
  - Bug 4: default model now imported from config.settings (single source of truth)
  - Bug 5: _require_key() raises EnvironmentError immediately on missing key,
           instead of passing "" to LangChain and producing a cryptic auth error.
"""
import os
from config.settings import LLM_MODEL as _DEFAULT_MODEL


def _require_key(env_var: str) -> str:
    """
    Return the value of env_var, stripped of whitespace.
    Raises EnvironmentError with a clear message if the key is absent or empty.
    """
    key = os.getenv(env_var, "").strip()
    if not key:
        raise EnvironmentError(
            f"'{env_var}' is not set. "
            f"Add it to your .env file or Streamlit secrets and restart the app."
        )
    return key


def get_llm(temperature: float = 0.3):
    """
    Return a LangChain chat model for the configured LLM_PROVIDER.
    Reads LLM_PROVIDER and LLM_MODEL from environment at call time.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model    = os.getenv("LLM_MODEL", _DEFAULT_MODEL)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=_require_key("GEMINI_API_KEY"),
            temperature=temperature,
            max_output_tokens=4096,
            convert_system_message_to_human=True,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            anthropic_api_key=_require_key("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=4096,
        )

    # Default: OpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        openai_api_key=_require_key("OPENAI_API_KEY"),
        temperature=temperature,
        max_tokens=4096,
    )
