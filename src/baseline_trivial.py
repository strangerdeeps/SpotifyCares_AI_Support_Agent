"""
baseline_trivial.py

The "trivial" baseline required by the assignment: pure keyword matching for
intent, canned template replies, and a fixed escalation rule. No LLM calls.
This exists purely as a floor -- the real system should beat this by a wide
margin, and by how much is part of the story.

Deliberately dumb:
- Intent: first keyword bucket that matches wins; otherwise "Non-Support/Noise".
- Reply: one fixed template per intent, no grounding in real history at all.
- Escalation: escalate only if a billing keyword AND a dispute keyword both
  appear; auto-handle everything else.
"""

INTENTS = [
    "Technical/Playback Issue",
    "Billing & Payment Issue",
    "Account Access Issue",
    "Subscription & Plan Management",
    "Content Availability/Catalog Complaint",
    "Feature Request/Product Feedback",
    "Non-Support/Noise",
]

KEYWORD_RULES = [
    ("Billing & Payment Issue", ["charge", "charged", "billing", "refund", "payment", "card", "promo", "$"]),
    ("Account Access Issue", ["log in", "login", "sign in", "password", "hacked", "someone else", "merged"]),
    ("Subscription & Plan Management", ["family", "student", "premium account", "plan", "subscription", "cancel"]),
    ("Content Availability/Catalog Complaint", ["not on spotify", "isn't on", "missing", "available in my country", "catalog", "album not"]),
    ("Feature Request/Product Feedback", ["would love", "feature", "suggestion", "please add", "wish"]),
    ("Technical/Playback Issue", ["crash", "freez", "wont play", "won't play", "bug", "not working", "stopped working", "glitch", "error", "won’t"]),
]

CANNED_REPLIES = {
    "Technical/Playback Issue": "Sorry to hear you're running into playback trouble! Please try restarting the app and your device, and let us know if the issue continues.",
    "Billing & Payment Issue": "We're sorry for the billing trouble. Please send us a DM with your account email so we can look into this.",
    "Account Access Issue": "Sorry you're having trouble accessing your account. Please try resetting your password, and DM us if that doesn't work.",
    "Subscription & Plan Management": "Thanks for reaching out about your subscription. Please DM us your account email and we'll take a look.",
    "Content Availability/Catalog Complaint": "Thanks for flagging this! Content availability depends on licensing agreements with rights holders, which we don't control directly.",
    "Feature Request/Product Feedback": "Thanks for the suggestion! We'll pass this along to our product team.",
    "Non-Support/Noise": "Thanks for reaching out!",
}

BILLING_DISPUTE_WORDS = ["dispute", "unauthorized", "didn't authorize", "without my permission", "fraud", "scam"]


def classify_intent(text: str) -> str:
    t = text.lower()
    for intent, keywords in KEYWORD_RULES:
        if any(k in t for k in keywords):
            return intent
    return "Non-Support/Noise"


def draft_reply(text: str, intent: str) -> str:
    return CANNED_REPLIES[intent]


def decide_escalation(text: str, intent: str) -> tuple[str, str]:
    t = text.lower()
    if intent == "Billing & Payment Issue" and any(w in t for w in BILLING_DISPUTE_WORDS):
        return "Escalate", "Billing dispute keyword detected (trivial rule)."
    return "Auto-handle", "No dispute keyword matched (trivial rule)."


def run(customer_text: str) -> dict:
    intent = classify_intent(customer_text)
    reply = draft_reply(customer_text, intent)
    escalate, reason = decide_escalation(customer_text, intent)
    return {
        "intent": intent,
        "reply": reply,
        "escalate": escalate,
        "escalate_reason": reason,
    }


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "I was charged twice this month, please help"
    print(run(text))
