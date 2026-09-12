"""
run_pipeline.py

One-call grounded SpotifyCares support agent.

The single LLM call performs:
1. Intent classification
2. Historical-resolution-grounded reply drafting
3. Escalation decision and reason

Designed to reduce Groq token usage while preserving
the assignment's required agent capabilities.
"""

from llm_client import call_llm_json
from retrieval import ThreadRetriever


_retriever = None


# IMPORTANT:
# These names must exactly match the golden-set labels.
INTENTS = [
    "Technical/Playback Issue",
    "Billing & Payment Issue",
    "Account Access Issue",
    "Subscription & Plan Management",
    "Content Availability/Catalog Complaint",
    "Feature Request/Product Feedback",
    "Non-Support/Noise",
]


# Compact definitions to reduce repeated input tokens.
INTENT_DEFINITIONS = """
Technical/Playback Issue: playback failures, crashes, freezing, app bugs.
Billing & Payment Issue: charges, refunds, payment failures, payment problems.
Account Access Issue: login, hacked or compromised accounts, ownership.
Subscription & Plan Management: Premium, Family, Student, plans, verification.
Content Availability/Catalog Complaint: missing songs, artists, albums, catalog.
Feature Request/Product Feedback: feature requests, suggestions, UI/UX feedback.
Non-Support/Noise: praise, spam, promotion, unrelated or non-actionable content.
""".strip()


def get_retriever():
    global _retriever

    if _retriever is None:
        _retriever = ThreadRetriever()

    return _retriever


def run(customer_message: str) -> dict:
    """
    Run the complete support agent for one customer message.
    """

    retriever = get_retriever()

    # One historical example keeps the request grounded
    # without unnecessarily increasing token usage.
    grounding = retriever.retrieve(
        customer_message,
        k=1,
        exclude_exact_match=True,
    )

    grounding_block = "No close historical example found."

    if grounding:
        item = grounding[0]

        past_message = str(
            item.get("past_customer_message", "")
        ).strip()

        past_reply = str(
            item.get("past_brand_reply", "")
        ).strip()

        if past_message and past_reply:
            grounding_block = (
                f"Customer: {past_message[:200]}\n"
                f"Spotify reply: {past_reply[:200]}"
            )

    prompt = f"""
You are SpotifyCares customer support.

Analyze the customer message and return ONLY valid JSON.

INTENT OPTIONS:
{INTENT_DEFINITIONS}

Choose exactly ONE intent from the options above.

ESCALATION:
Use "Escalate" if:
- a disputed or unauthorized charge/refund needs human action
- account security or compromise is involved
- persistent technical failure remains after troubleshooting
- account-specific action is required

Otherwise use "Auto-handle".

REPLY:
Write a helpful Spotify-style support reply under 280 characters.
Use the historical example for consistency.
Do not invent policies, refunds, account actions, or guarantees.

HISTORICAL EXAMPLE:
{grounding_block}

CUSTOMER MESSAGE:
{customer_message}

Return exactly these fields:
{{
  "intent": "one exact intent option",
  "reply": "support reply under 280 characters",
  "escalate": "Escalate or Auto-handle",
  "escalate_reason": "short explanation"
}}
""".strip()

    result = call_llm_json(
        "Return only the requested JSON object.",
        prompt,
        max_tokens=180,
    )

    # ---------------------------------------------------------
    # Validate intent
    # ---------------------------------------------------------

    intent = result.get("intent")

    if intent not in INTENTS:
        raise ValueError(
            f"Invalid intent returned: {result}"
        )

    # ---------------------------------------------------------
    # Validate escalation
    # ---------------------------------------------------------

    escalate = result.get("escalate")

    if escalate not in {
        "Escalate",
        "Auto-handle",
    }:
        raise ValueError(
            f"Invalid escalation value returned: {result}"
        )

    # ---------------------------------------------------------
    # Clean reply
    # ---------------------------------------------------------

    reply = str(
        result.get("reply", "")
    ).strip()

    # Ensure the final reply respects Twitter-style length.
    if len(reply) > 280:
        reply = reply[:277].rstrip() + "..."

    # ---------------------------------------------------------
    # Clean escalation reason
    # ---------------------------------------------------------

    escalate_reason = str(
        result.get("escalate_reason", "")
    ).strip()

    return {
        "customer_message": customer_message,

        "intent": intent,

        "intent_confidence": None,

        "reply": reply,

        "grounding_used": [
            {
                "past_message": g["past_customer_message"],
                "past_reply": g["past_brand_reply"],
                "similarity": g["similarity"],
            }
            for g in grounding
        ],

        "escalate": escalate,

        "escalate_reason": escalate_reason,
    }


if __name__ == "__main__":

    import sys
    import json

    text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "you charged me twice for my service this month"
    )

    result = run(text)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )