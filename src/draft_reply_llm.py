"""
draft_reply_llm.py

Drafts a reply grounded in how this brand has historically resolved similar
issues (assignment requirement #2), using retrieval.py to find real past
precedent rather than letting the LLM invent a plausible-sounding but
ungrounded response.
"""
from llm_client import call_llm
from retrieval import ThreadRetriever

SYSTEM_PROMPT = """You are drafting a customer support reply for Spotify's Twitter support
account (@SpotifyCares), in their voice: friendly, concise, casual-professional,
often signs off with initials (e.g. /RK). Twitter-length -- keep it under 280 characters.

You are given the current customer message plus 2-3 examples of how Spotify's
support team has actually replied to similar past issues. Use these as your
grounding for tone, structure, and typical resolution path (e.g. "DM us your
account email" is a very common real pattern for account-specific issues) --
don't just invent a generic reply; mirror what's actually worked before.

If the past examples show this type of issue typically requires collecting
account details via DM, do the same. If they show a direct fix is usually
suggested, suggest a similar fix.

Respond with ONLY the reply text, nothing else."""


def draft_reply(customer_message: str, retriever: ThreadRetriever, k: int = 3) -> dict:
    grounding = retriever.retrieve(customer_message, k=k)

    grounding_block = "\n\n".join(
        f"Past customer message: \"{g['past_customer_message']}\"\n"
        f"Past support reply: \"{g['past_brand_reply']}\""
        for g in grounding
    )

    user_prompt = f"""Current customer message: "{customer_message}"

Similar past issues and how support actually replied:
{grounding_block}

Draft the reply:"""

    reply = call_llm(SYSTEM_PROMPT, user_prompt, max_tokens=150)
    return {
        "reply": reply.strip(),
        "grounding_used": [
            {"past_message": g["past_customer_message"], "past_reply": g["past_brand_reply"], "similarity": g["similarity"]}
            for g in grounding
        ],
    }


if __name__ == "__main__":
    retriever = ThreadRetriever()
    result = draft_reply("my daily mixes never refresh, this has been happening for weeks", retriever)
    print("REPLY:", result["reply"])
    print("\nGROUNDED IN:")
    for g in result["grounding_used"]:
        print(f"  [{g['similarity']}] {g['past_message'][:80]} -> {g['past_reply'][:100]}")
