# Decision Log

Non-obvious engineering decisions made while building the SpotifyCares support agent, and why.

1. **Chose SpotifyCares over AmazonHelp/AppleSupport despite lower volume.**

   Spotify has a more bounded support domain centered on streaming, subscriptions,
   accounts, billing, and catalog issues. This made it possible to define a small,
   defensible intent taxonomy without creating product-specific categories for
   hundreds of unrelated products.

2. **Reconstructed complete SpotifyCares conversations instead of treating tweets independently.**

   The raw dataset contains threaded customer/support interactions. We reconstructed
   conversations using tweet relationships so retrieval and evaluation operate on
   actual support exchanges rather than isolated tweets.

   The final preprocessing produced 29,220 SpotifyCares threads.

3. **Shipped a targeted raw-data subsample instead of the full dataset.**

   The repository contains the raw tweets needed to reconstruct the SpotifyCares
   threads rather than the full 493MB dataset. This keeps the repository practical
   while preserving the complete set of tweets participating in the selected
   SpotifyCares conversations.

4. **Used seven intents rather than collapsing the taxonomy into four or five broad classes.**

   The final taxonomy separates Technical/Playback, Billing & Payment, Account Access,
   Subscription & Plan Management, Content Availability/Catalog Complaint,
   Feature Request/Product Feedback, and Non-Support/Noise.

   These distinctions matter because the appropriate support response differs
   substantially across these categories.

5. **Kept Non-Support/Noise as an explicit intent instead of filtering it out.**

   Real support inboxes contain praise, spam, promotion, and unrelated messages.
   Keeping this class tests whether the system can distinguish actionable support
   requests from messages that should not consume support effort.

6. **Used a single LLM call in the final grounded system for intent, reply, and escalation.**

   The final pipeline asks the model to produce the intent, customer-facing reply,
   escalation decision, and escalation reason together. This reduces API calls,
   latency, and opportunities for inconsistent decisions between separate models or
   prompts.

   The escalation prompt explicitly identifies high-risk cases such as unauthorized
   charges, account compromise, persistent failures, and account-specific actions.

7. **Created a 220-example manually reviewed golden set.**

   A fixed evaluation set was created so that every system and baseline is compared
   on exactly the same examples. The labels were manually reviewed after an initial
   quality check identified overuse of the Non-Support/Noise class.

8. **Performed a second labeling QA pass instead of accepting the first labeling pass.**

   The first pass contained 87/220 Non-Support/Noise labels. Reviewing the relationship
   between those labels and the rough heuristic revealed that some substantive support
   requests had been incorrectly grouped as noise.

   The revised set contains 35/220 Non-Support/Noise examples and provides a more
   defensible evaluation taxonomy.

9. **Used TF-IDF retrieval over historical customer messages for grounding.**

   Retrieval is performed locally over the 29,220 reconstructed SpotifyCares threads.
   TF-IDF was chosen because it is inexpensive, deterministic, transparent, and does
   not require another API or embedding service.

   The tradeoff is that lexical retrieval can miss semantically similar messages
   expressed with different vocabulary.

10. **Used only the closest historical example in the final grounded pipeline.**

    The final pipeline retrieves the top historical match and gives the customer
    message plus that historical customer/reply pair to the LLM.

    This keeps the prompt compact and reduces API token usage under the free-plan
    rate limits while still providing concrete historical grounding.

11. **Compared against both a trivial baseline and a zero-shot LLM baseline.**

    The trivial baseline establishes how far a very simple deterministic approach can
    get. The zero-shot baseline tests whether retrieval and the additional system
    structure add value beyond simply asking an LLM to classify the message.

    This makes the comparison more informative than comparing only against another
    LLM configuration.

12. **Evaluated escalation using the Escalate class specifically rather than relying on accuracy.**

    The golden set contains 179 Auto-handle and 41 Escalate examples. Because of this
    imbalance, escalation accuracy can look strong even when a system misses many
    cases requiring human intervention.

    Escalate precision, recall, and F1 are therefore reported, with particular
    attention to Escalate F1 and Escalate Recall.

13. **Used four independent dimensions for reply-quality evaluation.**

    The LLM judge scores groundedness, correctness, tone fit, and actionability
    separately rather than producing only one overall score.

    This makes failure analysis more useful because a reply can be factually correct
    but poorly actionable, or well-written but insufficiently grounded.

14. **Used a 35-example subset for judge-agreement evaluation.**

    The same fixed subset is scored by the LLM judge and independently reviewed by a
    human. Agreement is measured using Spearman correlation, exact-match rate, and
    within-one-point agreement for each rubric dimension.

    The subset is small enough to review manually while still exposing disagreement
    patterns in the judge.

15. **Used Groq with Qwen 3.6 27B for the final experiments.**

    The original Anthropic API was unavailable for continued experimentation because
    the account had insufficient API credit. The provider was therefore switched to
    Groq using the free-plan-compatible Qwen 3.6 27B model.

    The LLM provider is isolated in `src/llm_client.py`, so the retrieval,
    evaluation, and pipeline code remains provider-independent. Bounded retry and
    throttling logic were added to handle rate limits.
