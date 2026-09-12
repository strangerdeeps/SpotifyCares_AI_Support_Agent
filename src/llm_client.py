"""
llm_client.py

Thin wrapper around Groq's free-plan chat-completions API.

Features:
- Proactive throttling for Groq's request-per-minute limit.
- Automatic retry for temporary 429/5xx errors.
- Immediate stop for daily quota exhaustion.
- JSON response support.
"""

import os
import json
import time

import requests


MODEL = os.environ.get(
    "HIVER_MODEL",
    "qwen/qwen3.6-27b",
)

API_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)


# Keep a safety margin below Groq's 30 RPM limit.
MIN_SECONDS_BETWEEN_CALLS = 2.4


# Time of the previous API request.
_last_call_time = 0.0


def _get_api_key() -> str:
    """Get the Groq API key from the environment."""

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. "
            "Get a key at https://console.groq.com/keys "
            "and set GROQ_API_KEY in your shell."
        )

    return api_key.strip()


def _wait_for_throttle():
    """
    Wait long enough to stay below the requests-per-minute
    limit.
    """

    global _last_call_time

    now = time.monotonic()
    elapsed = now - _last_call_time

    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        wait_s = MIN_SECONDS_BETWEEN_CALLS - elapsed
        time.sleep(wait_s)


def _is_daily_quota_error(response) -> bool:
    """
    Detect whether a 429 response is caused by a daily
    request/token quota rather than a temporary rate limit.
    """

    text = response.text.lower()

    daily_indicators = [
        "tokens per day",
        "token per day",
        "tokens/day",
        "token/day",
        "tpd",
        "requests per day",
        "request per day",
        "requests/day",
        "request/day",
        "rpd",
        "daily limit",
        "daily quota",
        "quota exceeded",
    ]

    return any(
        indicator in text
        for indicator in daily_indicators
    )


def _retry_delay(response, attempt: int) -> float:
    """
    Determine how long to wait before retrying a temporary
    429/5xx response.
    """

    retry_after = response.headers.get(
        "retry-after"
    )

    try:
        if retry_after:
            wait_s = float(retry_after)
        else:
            wait_s = 2 ** attempt

    except (TypeError, ValueError):
        wait_s = 2 ** attempt

    return min(
        max(wait_s, 1.0),
        30.0,
    )


def call_llm(
    system: str,
    user: str,
    max_tokens: int = 1600,
    temperature: float = 0.6,
    json_mode: bool = False,
) -> str:
    """
    Call Groq's chat-completions API.

    Qwen 3.6 uses no additional reasoning effort here.
    This keeps the response focused and reduces unnecessary
    token consumption.
    """

    global _last_call_time

    headers = {
        "Authorization": (
            f"Bearer {_get_api_key()}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,

        # Qwen 3.6 supports "none" / "default".
        "reasoning_effort": "none",

        "include_reasoning": False,

        "messages": [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
    }

    if json_mode:
        payload["response_format"] = {
            "type": "json_object"
        }

    max_retries = 5

    for attempt in range(max_retries + 1):

        # -----------------------------------------------------
        # Proactive RPM throttling
        # -----------------------------------------------------

        _wait_for_throttle()

        try:
            resp = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )

            # Record when the request was actually sent.
            _last_call_time = time.monotonic()

        except requests.RequestException as exc:

            _last_call_time = time.monotonic()

            if attempt == max_retries:
                raise

            wait_s = min(
                2 ** attempt,
                30,
            )

            print(
                f"Request error: {exc}. "
                f"Retrying in {wait_s}s..."
            )

            time.sleep(wait_s)
            continue

        # -----------------------------------------------------
        # 429 Rate Limit
        # -----------------------------------------------------

        if resp.status_code == 429:

            # Daily quota errors should NOT be retried.
            if _is_daily_quota_error(resp):

                print()
                print(
                    "===== GROQ DAILY QUOTA REACHED ====="
                )
                print(
                    "Status:",
                    resp.status_code,
                )
                print(
                    "Response:",
                    resp.text,
                )
                print(
                    "===================================="
                )
                print()

                raise RuntimeError(
                    "Groq daily quota reached. "
                    "The successfully processed rows "
                    "have already been checkpointed. "
                    "Wait for the quota to reset and "
                    "run the evaluation again."
                )

            # Temporary RPM/TPM limit.
            if attempt == max_retries:

                print(
                    "GROQ ERROR:",
                    resp.status_code,
                    resp.text,
                )

                resp.raise_for_status()

            wait_s = _retry_delay(
                resp,
                attempt,
            )

            print(
                f"Groq returned 429. "
                f"Retrying in {wait_s:.1f}s..."
            )

            time.sleep(wait_s)
            continue

        # -----------------------------------------------------
        # Server errors
        # -----------------------------------------------------

        if 500 <= resp.status_code < 600:

            if attempt == max_retries:

                print(
                    "GROQ ERROR:",
                    resp.status_code,
                    resp.text,
                )

                resp.raise_for_status()

            wait_s = min(
                2 ** attempt,
                30,
            )

            print(
                f"Groq returned "
                f"{resp.status_code}. "
                f"Retrying in {wait_s}s..."
            )

            time.sleep(wait_s)
            continue

        # -----------------------------------------------------
        # Other HTTP errors
        # -----------------------------------------------------

        if not resp.ok:

            print()
            print(
                "===== GROQ API ERROR ====="
            )
            print(
                "Status:",
                resp.status_code,
            )
            print(
                "Response:",
                resp.text,
            )
            print(
                "=========================="
            )
            print()

            resp.raise_for_status()

        # -----------------------------------------------------
        # Parse successful response
        # -----------------------------------------------------

        data = resp.json()

        try:
            return data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:

            raise RuntimeError(
                "Unexpected Groq response format:\n"
                f"{data}"
            ) from exc

    raise RuntimeError(
        "LLM request failed after all retries."
    )


def call_llm_json(
    system: str,
    user: str,
    max_tokens: int = 1600,
) -> dict:
    """
    Call the LLM and parse its JSON response.
    """

    raw = call_llm(
        system,
        user,
        max_tokens=max_tokens,
        temperature=0.6,
        json_mode=True,
    )

    cleaned = raw.strip()

    # Handle accidental Markdown code fences.
    if cleaned.startswith("```"):

        lines = cleaned.splitlines()

        if lines and lines[0].strip().lower() in {
            "```json",
            "```",
        }:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

    try:

        result = json.loads(cleaned)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Groq returned invalid JSON.\n\n"
            f"Raw response:\n{raw}"
        ) from exc

    if not isinstance(result, dict):

        raise RuntimeError(
            "Groq JSON response was not an object.\n\n"
            f"Raw response:\n{raw}"
        )

    return result