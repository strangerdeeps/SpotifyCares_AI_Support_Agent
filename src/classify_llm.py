"""
classify_llm.py

Few-shot LLM intent classifier. Exemplars are hand-picked from the full
29k-thread corpus and verified to have zero overlap with the golden
evaluation set (see data/fewshot_exemplars.json and the check that produced
it) -- using golden-set examples as few-shot exemplars would let the
classifier "see" eval answers indirectly and invalidate the accuracy numbers.
"""
import json
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

with open("data/fewshot_exemplars.json") as f:
    EXEMPLARS = json.load(f)

INTENT_DEFINITIONS = """
- Technical/Playback Issue: app crashes, freezing, songs won't play, sync bugs across devices
- Billing & Payment Issue: unexpected charges, failed payments, promo codes not applying, refund requests
- Account Access Issue: can't log in, account compromised, duplicate/merged accounts
- Subscription & Plan Management: family plan setup, student verification, tier showing wrong (Free vs Premium)
- Content Availability/Catalog Complaint: missing songs/artists/albums -- not resolvable by support, just acknowledgeable
- Feature Request/Product Feedback: UI/UX suggestions, not a problem to fix
- Non-Support/Noise: fan praise, spam/promo tweets, unrelated chatter -- not an actionable support request
""".strip()


def build_fewshot_block() -> str:
    lines = []
    for intent, examples in EXEMPLARS.items():
        for ex in examples:
            lines.append(f'Message: "{ex}"\nIntent: {intent}\n')
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are an intent classifier for Spotify's customer support Twitter account.
Classify each customer message into exactly one of these intents:
{INTENT_DEFINITIONS}

Examples:
{build_fewshot_block()}

Respond with ONLY a JSON object: {{"intent": "<one of the exact intent names above>", "confidence": <0-1 float>}}
No other text."""


def classify(customer_message: str) -> dict:
    result = call_llm_json(SYSTEM_PROMPT, f'Message: "{customer_message}"\nIntent:')
    if result.get("intent") not in INTENTS:
        # Fail loud rather than silently mislabeling -- caller can decide fallback policy
        raise ValueError(f"LLM returned invalid intent: {result}")
    return result


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "you charged me twice for my service this month"
    print(classify(text))
