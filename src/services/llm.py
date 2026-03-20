import asyncio
import json

import litellm
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

litellm.drop_params = True


async def complete_json(
    system_prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int | None = None,
) -> dict:
    _model = model or settings.EXTRACT_MODEL
    _temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
    _retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES

    kwargs: dict = {
        "model": _model,
        "messages": [{"role": "system", "content": system_prompt}],
        "response_format": {"type": "json_object"},
        "temperature": _temperature,
    }
    if settings.LLM_API_KEY:
        kwargs["api_key"] = settings.LLM_API_KEY
    if settings.LLM_BASE_URL:
        kwargs["api_base"] = settings.LLM_BASE_URL

    last_exc: Exception | None = None
    for attempt in range(1, _retries + 1):
        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("llm_json_parse_error", attempt=attempt, error=str(e))
            last_exc = e
        except Exception as e:
            logger.warning("llm_completion_error", attempt=attempt, error=str(e))
            last_exc = e
        if attempt < _retries:
            await asyncio.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"LLM call failed after {_retries} attempts: {last_exc}")
