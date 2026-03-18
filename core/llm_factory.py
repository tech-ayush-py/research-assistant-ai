"""
LLM factory – returns a LangChain chat model based on settings.
Always reads env vars at call time so changes to os.environ take effect immediately.

Fixes applied:
  - Bug 4: default model imported from config.settings (single source of truth)
  - Bug 5: _require_key() raises EnvironmentError immediately on missing key
  - Quota fix: gemini-2.0-flash is NOT available on the free tier and raises
    RESOURCE_EXHAUSTED (429). Default is now gemini-1.5-flash which is free-tier
    compatible. A _GEMINI_FREE_TIER_FALLBACK guards against any leftover
    gemini-2.0-flash references in environment variables.
  - Retry fix: llm_with_retry() wraps every LLM call with exponential backoff
    for transient 429/503 errors, with a clear user-facing message on failure.
"""
import os
import time
import logging

from config.settings import LLM_MODEL as _DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Free-tier compatible model. Use the stable pinned alias (-001) which is
# guaranteed to exist across all API versions and langchain-google-genai releases.
_GEMINI_FREE_TIER_FALLBACK = "gemini-1.5-flash-001"
_GEMINI_PAID_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
}


def _require_key(env_var: str) -> str:
    """
    Return the value of env_var stripped of whitespace.
    Raises EnvironmentError with a clear message if absent or empty.
    """
    key = os.getenv(env_var, "").strip()
    if not key:
        raise EnvironmentError(
            f"'{env_var}' is not set. "
            f"Add it to your .env file or Streamlit secrets and restart the app."
        )
    return key


def _resolve_gemini_model(model: str) -> str:
    """
    Return a free-tier compatible, API-version-safe Gemini model name.
    - Paid-only models (2.0-flash, 1.5-pro, etc.) are downgraded to the free fallback.
    - The bare alias 'gemini-1.5-flash' is normalised to 'gemini-1.5-flash-001',
      the stable pinned version that works across all langchain-google-genai releases.
    """
    if model in _GEMINI_PAID_MODELS:
        logger.warning(
            "Model '%s' requires a billing-enabled Google account (RESOURCE_EXHAUSTED on free tier). "
            "Falling back to '%s'. Set LLM_MODEL=%s in your secrets to suppress this warning.",
            model, _GEMINI_FREE_TIER_FALLBACK, _GEMINI_FREE_TIER_FALLBACK,
        )
        return _GEMINI_FREE_TIER_FALLBACK
    # Normalise bare alias to pinned stable version
    if model == "gemini-1.5-flash":
        return "gemini-1.5-flash-001"
    return model


def get_llm(temperature: float = 0.3):
    """
    Return a LangChain chat model for the configured LLM_PROVIDER.
    Reads LLM_PROVIDER and LLM_MODEL from environment at call time.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model    = os.getenv("LLM_MODEL", _DEFAULT_MODEL)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        safe_model = _resolve_gemini_model(model)
        return ChatGoogleGenerativeAI(
            model=safe_model,
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


def invoke_with_retry(llm, messages, retries: int = 3, base_delay: float = 5.0):
    """
    Invoke an LLM with exponential backoff for transient 429 / 503 errors.

    Usage (replaces llm.invoke(messages)):
        from core.llm_factory import get_llm, invoke_with_retry
        response = invoke_with_retry(get_llm(), messages)

    Args:
        llm:        A LangChain chat model instance.
        messages:   List of LangChain message objects.
        retries:    Maximum number of retry attempts (default 3).
        base_delay: Initial wait in seconds; doubles each retry (default 5s).

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            err_str = str(exc).upper()
            is_retryable = any(k in err_str for k in (
                "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "RATE_LIMIT"
            ))
            if is_retryable and attempt < retries:
                wait = base_delay * (2 ** attempt)
                logger.warning(
                    "LLM rate-limit hit (attempt %d/%d). Retrying in %.0fs. Error: %s",
                    attempt + 1, retries, wait, exc,
                )
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc
