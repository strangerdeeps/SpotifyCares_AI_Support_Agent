"""
retrieval.py

Grounds the reply-drafting step in how this brand has actually resolved
similar issues before, as required by the assignment ("Draft a reply
grounded in how that brand has historically resolved similar issues").

Approach: TF-IDF similarity over customer openers in the full thread corpus
(29,220 SpotifyCares conversations). Given a new customer message, retrieve
the top-k most similar past conversations and return their full thread
(including the brand's actual historical reply) as grounding context for
the reply drafter.

Why TF-IDF and not embeddings: no network-dependent embedding API call is
needed, it's fast enough to run over ~30k threads in-process, and it's
transparent/debuggable -- you can see exactly why two messages matched.
Documented as a decision-log tradeoff: embeddings would likely retrieve
better semantic (non-lexical) matches, at the cost of an extra API
dependency and latency. Worth trying as a "next week" improvement.
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ThreadRetriever:
    def __init__(self, threads_path: str = "data/spotify_threads.json"):
        with open(threads_path) as f:
            self.threads = json.load(f)

        # Index on the customer's opening message of each thread
        self.openers = [t[0]["text"] for t in self.threads]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=20000, ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform(self.openers)

    def retrieve(self, query: str, k: int = 3, exclude_exact_match: bool = True):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = sims.argsort()[::-1]

        results = []
        for idx in ranked_idx:
            if exclude_exact_match and self.openers[idx].strip() == query.strip():
                continue
            if sims[idx] <= 0:
                break
            thread = self.threads[idx]
            brand_reply = next((t["text"] for t in reversed(thread) if not t["inbound"]), None)
            results.append(
                {
                    "similarity": round(float(sims[idx]), 4),
                    "past_customer_message": thread[0]["text"],
                    "past_brand_reply": brand_reply,
                    "full_thread": thread,
                }
            )
            if len(results) >= k:
                break
        return results


if __name__ == "__main__":
    retriever = ThreadRetriever()
    query = "my daily mixes never refresh, please fix this bug"
    results = retriever.retrieve(query, k=3)
    print(f"Query: {query}\n")
    for r in results:
        print(f"[sim={r['similarity']}] Past customer: {r['past_customer_message'][:100]}")
        print(f"           -> Brand replied: {r['past_brand_reply'][:150]}")
        print()
