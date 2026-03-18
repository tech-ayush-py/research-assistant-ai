"""
LLM factory — returns a callable LLM object for the configured provider.

KEY CHANGE: For Gemini, we now call google-generativeai directly instead of
going through langchain-google-genai. The LangChain wrapper hardcodes the
v1beta endpoint in older releases (1.x), which rejects valid model names with
a 404 NOT_FOUND. The google-generativeai SDK always uses the correct endpoint.

The returned object exposes a single method:
    response = llm.invoke(messages)   # messages = list of LangChain message objects
    text = response.content           # plain string

This is 100% compatible with all existing agent code — no agent changes needed.
"""
import os
import time
import logging

from config.settings import LLM_MODEL as _DEFAULT_MODEL

logger = logging.getLogger(__name__)

# gemini-2.0-* and gemini-1.5-pro require a paid account.
# gemini-1.5-flash is the correct free-tier model.
_GEMINI_FREE_TIER_MODEL = "gemini-1.5-flash"
_GEMINI_PAID_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
}


def _require_key(env_var: str) -> str:
    key = os.getenv(env_var, "").strip()
    if not key:
        raise EnvironmentError(
            f"'{env_var}' is not set. "
            f"Add it to your .env file or Streamlit secrets and restart the app."
        )
    return key


def _resolve_gemini_model(model: str) -> str:
    """Return a free-tier compatible model name for the google-generativeai SDK."""
    # Normalise the pinned alias back to the bare name the SDK accepts
    if model == "gemini-1.5-flash-001":
        return _GEMINI_FREE_TIER_MODEL
    if model in _GEMINI_PAID_MODELS:
        logger.warning(
            "Model '%s' requires a paid Google account. Falling back to '%s'.",
            model, _GEMINI_FREE_TIER_MODEL,
        )
        return _GEMINI_FREE_TIER_MODEL
    return model


class _Response:
    """Minimal response object — exposes .content like LangChain AIMessage."""
    def __init__(self, text: str):
        self.content = text


class _GeminiDirectLLM:
    """
    Calls google-generativeai SDK directly — bypasses langchain-google-genai
    entirely so there is no v1beta / v1 endpoint confusion.
    Interface is identical to a LangChain chat model: .invoke(messages) -> _Response.
    """
    def __init__(self, model: str, api_key: str, temperature: float):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model_name = model
        self._temperature = temperature
        self._genai = genai

    def invoke(self, messages) -> _Response:
        # Convert list of LangChain message objects to a single prompt string
        parts = []
        for msg in messages:
            text = getattr(msg, "content", str(msg))
            parts.append(text)
        prompt = "\n\n".join(parts)

        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=self._genai.types.GenerationConfig(
                temperature=self._temperature,
                max_output_tokens=4096,
            ),
        )
        resp = model.generate_content(prompt)
        return _Response(resp.text)


def get_llm(temperature: float = 0.3):
    """
    Return an LLM object for the configured LLM_PROVIDER.
    Always reads env vars at call time.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model    = os.getenv("LLM_MODEL", _DEFAULT_MODEL)

    if provider == "gemini":
        safe_model = _resolve_gemini_model(model)
        return _GeminiDirectLLM(
            model=safe_model,
            api_key=_require_key("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            anthropic_api_key=_require_key("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=4096,
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        openai_api_key=_require_key("OPENAI_API_KEY"),
        temperature=temperature,
        max_tokens=4096,
    )


def invoke_with_retry(llm, messages, retries: int = 3, base_delay: float = 5.0):
    """
    Call llm.invoke(messages) with exponential backoff on 429 / 503 errors.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            err_str = str(exc).upper()
            is_retryable = any(k in err_str for k in (
                "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "RATE_LIMIT",
            ))
            if is_retryable and attempt < retries:
                wait = base_delay * (2 ** attempt)
                logger.warning(
                    "LLM rate-limit hit (attempt %d/%d). Retrying in %.0fs.",
                    attempt + 1, retries, wait,
                )
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc
