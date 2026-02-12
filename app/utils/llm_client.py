import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# LLM provider: "ollama" (default, local, private) or "groq" (cloud, opt-in)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# ollama config (local - your data never leaves your machine)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# groq config (cloud - WARNING: data is sent to Groq's servers)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2048


def _generate_ollama(messages: list, temperature: float, max_tokens: int) -> str:
    import ollama
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    msg = response.message
    return msg.content if hasattr(msg, 'content') else msg.get('content', '')


def _generate_groq(messages: list, temperature: float, max_tokens: int) -> str:
    from groq import Groq
    if not GROQ_API_KEY:
        raise ConnectionError("GROQ_API_KEY not set. Set it in .env or use LLM_PROVIDER=ollama for local inference.")
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def generate(prompt: str, system: Optional[str] = None,
             temperature: float = DEFAULT_TEMPERATURE,
             max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Generate text from the LLM with retry logic. Supports ollama (local) and groq (cloud)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    provider = LLM_PROVIDER.lower()
    gen_fn = _generate_groq if provider == "groq" else _generate_ollama

    for attempt in range(MAX_RETRIES):
        try:
            return gen_fn(messages, temperature, max_tokens)
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}, provider={provider}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"LLM call failed after {MAX_RETRIES} attempts")
                raise ConnectionError(f"LLM ({provider}) unreachable: {e}")

    return ""


def generate_json(prompt: str, system: Optional[str] = None,
                  temperature: float = 0.1,
                  max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """Generate structured JSON output from the LLM."""
    json_system = (system or "") + "\nRespond ONLY with valid JSON. No markdown, no explanation."

    raw = generate(prompt, system=json_system, temperature=temperature, max_tokens=max_tokens)

    # try to extract JSON from the response
    text = raw.strip()

    # handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse LLM response as JSON: {text[:200]}")
        return {"raw_response": raw, "parse_error": True}


def check_connection() -> bool:
    """Check if the configured LLM provider is reachable."""
    provider = LLM_PROVIDER.lower()

    if provider == "groq":
        try:
            from groq import Groq
            if not GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not set")
                return False
            client = Groq(api_key=GROQ_API_KEY)
            client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            logger.info(f"Groq connected. Model: {GROQ_MODEL}")
            return True
        except Exception as e:
            logger.error(f"Groq connection failed: {e}")
            return False

    # default: ollama
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_HOST)
        models = client.list()
        model_list = getattr(models, 'models', None) or []
        available = [getattr(m, 'model', str(m)) for m in model_list]
        if any(OLLAMA_MODEL in m for m in available):
            logger.info(f"Ollama connected. Model {OLLAMA_MODEL} available.")
            return True
        else:
            logger.warning(f"Model {OLLAMA_MODEL} not found. Available: {available}")
            return False
    except Exception as e:
        logger.error(f"Cannot connect to Ollama at {OLLAMA_HOST}: {e}")
        return False


def get_provider_info() -> dict:
    """Return current LLM config for the frontend."""
    provider = LLM_PROVIDER.lower()
    if provider == "groq":
        return {
            "provider": "groq",
            "model": GROQ_MODEL,
            "privacy": "cloud - data is sent to Groq servers",
        }
    return {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "privacy": "local - your data never leaves this machine",
    }
