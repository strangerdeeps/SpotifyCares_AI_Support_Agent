"""
baseline_zeroshot_llm.py

The "simple" baseline required by the assignment (distinct from the trivial
keyword baseline): one zero-shot LLM call does classification, reply
drafting, and escalation decision all at once, with NO few-shot examples,
NO retrieval/grounding in historical replies, and NO hybrid escalation
policy. This isolates exactly what retrieval + few-shot + policy rules
(the full system) add over "just ask an LLM."
"""
from llm_client import call_llm_json

INTENTS = [
    "Technical/Playback Issue",
    "Billing & Payment Issue",
    "Account Access Issue",
    "Subscription & Plan Management",
    "Content Availability/Catalog Complaint",
    "Feature Request/Product Feedback",
    "Non-Support/Noise",
]

SYSTEM_PROMPT = f"""You are Spotify's customer support Twitter agent. Given a customer
message, do three things:
1. Classify its intent as exactly one of: {', '.join(INTENTS)}
2. Draft a brief support reply (under 280 characters).
3. Decide "Escalate" or "Auto-handle" and give a one-sentence reason.

Respond with ONLY a JSON object:
{{"intent": "...", "reply": "...", "escalate": "Escalate"|"Auto-handle", "escalate_reason": "..."}}
No other text."""


def run(customer_message: str) -> dict:
    return call_llm_json(SYSTEM_PROMPT, f'Customer message: "{customer_message}"', max_tokens=400)


if __name__ == "__main__":
    print(run("you charged me twice for my service this month"))
