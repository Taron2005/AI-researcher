"""
Thin wrapper around the OpenRouter API, shared by all three agent roles.

OpenRouter speaks the same chat-completions format as OpenAI's API, so we use
the official `openai` SDK pointed at OpenRouter's base URL instead of hand-
rolling HTTP requests and response parsing — this is the standard client for
this API shape, not an extra abstraction (CLAUDE.md rule 4).

Each role module (harness/roles/*.py) calls `call_model()` with its own model
ID, temperature, and messages/tools — this file has no knowledge of what a
"Planner" or "Reviewer" is, it only knows how to make one API call.
"""

import os
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI

from harness.config import OPENROUTER_BASE_URL
from harness.trace import Trace

load_dotenv()

_api_key = os.environ.get("OPENROUTER_API_KEY")
if not _api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
    )

_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=_api_key)

# Only genuinely transient failures -- a rate limit, a network blip, a
# provider-side 5xx. Deliberately NOT the broader `openai.APIStatusError`
# base class, which also covers permanent errors (401 bad key, 402
# insufficient credits -- hit for real earlier in this project) that
# retrying can never fix, only delay noticing.
_RETRYABLE_ERRORS = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
_MAX_RETRIES = 3


def _call_with_retry(**kwargs):
    for attempt in range(_MAX_RETRIES):
        try:
            return _client.chat.completions.create(**kwargs)
        except _RETRYABLE_ERRORS:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)  # 1s, 2s, 4s


def call_model(
    model: str,
    messages: list[dict],
    temperature: float,
    tools: list[dict] | None = None,
    online: bool = False,
    trace: Trace | None = None,
    stage: str = "unknown",
):
    """
    Make one chat-completion call and return the raw response object.

    Callers read `.choices[0].message.content` for plain text, or
    `.choices[0].message.tool_calls` when `tools` was passed and the model
    chose to call one.

    `online=True` enables OpenRouter's built-in web search for this one call
    (appends ":online" to the model ID) — see DECISIONS.md for why this
    replaces a custom web_search tool. It's a per-call setting, not a tool
    the model picks from `tools`, because OpenRouter implements it as a
    model/request-level feature, not a function call.

    If `trace` is given, this call logs itself (model, latency, tokens,
    cost, and whether a tool was called) under `stage` — logging lives here,
    once, instead of being repeated in every role module that calls this.
    """
    effective_model = f"{model}:online" if online else model
    kwargs = {"model": effective_model, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools

    start = time.monotonic()
    response = _call_with_retry(**kwargs)
    latency_s = time.monotonic() - start

    if trace is not None:
        message = response.choices[0].message
        trace.log_event(
            stage=stage,
            event_type="llm_call",
            model=effective_model,
            temperature=temperature,
            latency_s=round(latency_s, 2),
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            cost=response.usage.cost if response.usage else None,
            tool_calls=[tc.function.name for tc in (message.tool_calls or [])],
            reply_preview=(message.content or "")[:200],
        )

    return response
