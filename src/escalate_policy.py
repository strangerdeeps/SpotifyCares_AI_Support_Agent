"""
escalate_policy.py

Decides auto-handle vs escalate, with a stated reason (assignment
requirement #3). Hybrid design, not pure LLM judgment call:

1. Hard policy rules based on intent (fast, auditable, no LLM needed for the
   clear-cut cases): Feature Request and Non-Support/Noise NEVER escalate --
   there's no account action to take either way. This matches what we saw
   in the golden set (0 escalations in either category).

2. For the intents where real judgment is needed (Billing, Account Access,
   Subscription, Technical), ask the LLM for a risk signal grounded in
   specific red flags (financial exposure, security/identity risk,
   signs of repeated/unresolved issue) rather than a bare yes/no -- this
   is what makes the "stated reason" meaningful rather than decorative.

Why not pure LLM end to end: the golden set shows Feature Request and Noise
should NEVER escalate (0/38 and 0/35 respectively) -- a policy floor here
is cheaper, faster, and more auditable than trusting the LLM to reliably
learn "never escalate this category" from a prompt alone.
"""
from llm_client import call_llm_json

NEVER_ESCALATE_INTENTS = {"Feature Request/Product Feedback", "Non-Support/Noise"}

SYSTEM_PROMPT = """You are deciding whether a Spotify customer support message needs to be
escalated to a human agent, or can be auto-handled with a standard reply.

Escalate if ANY of these apply:
- Real financial exposure (a specific disputed charge, refund request, "unauthorized" or "fraud")
- Security/identity risk (possible account compromise, someone else has access)
- The customer describes repeated/persistent failure of standard troubleshooting
- Resolving it requires account-specific data a general reply can't provide

Otherwise, auto-handle (general guidance, standard troubleshooting, or a
question with a public, non-account-specific answer suffices).

Respond with ONLY a JSON object:
{"escalate": "Escalate" or "Auto-handle", "reason": "<one sentence citing the specific signal in THIS message>"}
No other text."""


def decide(customer_message: str, intent: str) -> dict:
    if intent in NEVER_ESCALATE_INTENTS:
        return {
            "escalate": "Auto-handle",
            "reason": f"Policy: {intent} is never escalated -- no account-specific action is possible either way.",
        }
    result = call_llm_json(
        SYSTEM_PROMPT, f'Intent: {intent}\nMessage: "{customer_message}"\nDecision:'
    )
    return result


if __name__ == "__main__":
    print(decide("you charged me twice this month, I never approved this", "Billing & Payment Issue"))
