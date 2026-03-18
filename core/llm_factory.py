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

# Models that require a paid/billing-enabled account — downgrade to free tier automatically.
# gemini-2.5-pro is intentionally NOT in this list: it has a limited free tier
# (25 req/day, 5 RPM) and is the user-requested model.
_GEMINI_FREE_TIER_MODEL = "gemini-1.5-flash"   # fallback if a blocked model is requested
_GEMINI_PAID_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-preview",
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


def _parse_retry_delay(err_str: str) -> float:
    """
    Extract the retry_delay seconds from a Gemini 429 error message.
    Falls back to base_delay if not found.
    """
    import re
    match = re.search(r"retry_delay['\"]?\s*[:{]\s*['\"]?seconds['\"]?\s*[:=]\s*(\d+)", err_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"retry.in.(\d+).second", err_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def invoke_with_retry(llm, messages, retries: int = 3, base_delay: float = 5.0):
    """
    Call llm.invoke(messages) with smart retry logic:
    - Parses the retry_delay from Gemini 429 responses and waits exactly that long
    - Distinguishes per-minute rate limits (retryable) from daily quota exhaustion (not retryable)
    - Raises a clear QuotaExhaustedError for daily quota so the app can show a helpful message
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            err_str = str(exc)
            err_upper = err_str.upper()

            is_rate_limited = any(k in err_upper for k in (
                "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "RATE_LIMIT",
            ))

            # Daily quota exhausted — not retryable, fail immediately with clear message
            is_daily_quota = any(k in err_upper for k in (
                "PERDAY", "PER_DAY", "DAILY", "DAY_LIMIT",
            ))
            if is_daily_quota or (is_rate_limited and "FREEDAY" in err_upper.replace("_", "").replace("-", "")):
                raise RuntimeError(
                    "Daily free-tier quota exhausted for this model. "
                    "You have used all free requests for today. "
                    "Options: (1) wait until midnight PT for the quota to reset, "
                    "(2) switch to a model with a higher daily limit, "
                    "or (3) add billing to your Google AI account."
                ) from exc

            if is_rate_limited and attempt < retries:
                # Honour the retry_delay from the API response if present
                suggested = _parse_retry_delay(err_str)
                wait = suggested if suggested else base_delay * (2 ** attempt)
                logger.warning(
                    "LLM rate-limit hit (attempt %d/%d). Waiting %.0fs as suggested by API.",
                    attempt + 1, retries, wait,
                )
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc
