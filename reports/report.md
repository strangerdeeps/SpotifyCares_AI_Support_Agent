# Report: AI Support Agent for @SpotifyCares

## 1. Problem framing

### Brand

I chose SpotifyCares because it has a bounded support domain: playback, accounts, subscriptions, payments, catalog/content, and product feedback. This makes a small, defensible intent taxonomy possible without collapsing materially different support workflows.

### What "good" means

- **Intent classification:** route a message according to the support action it requires, not just topical similarity. For example, a playback failure should be troubleshot, while a feature request should be recorded rather than treated as a bug.
- **Reply quality:** stay grounded in historical SpotifyCares resolutions. A plausible generic answer is not sufficient if it invents a policy, product path, refund, or action.
- **Escalation:** reliably identify cases where human attention is important, especially disputed/unauthorized payments, account security, and persistent unresolved problems, while avoiding unnecessary escalation of routine feedback or noise.

### What I chose not to build

- **Multi-language support:** non-English messages exist in the corpus, but this prototype focuses on the dominant support workflow.
- **A dedicated spam pre-filter:** promotional/noise messages are represented by the `Non-Support/Noise` intent for this prototype. A production system would filter these earlier.
- **Embedding retrieval:** I used TF-IDF retrieval because it is local, deterministic, inexpensive, and easy to debug. Embeddings are a planned comparison.
- **Full multi-turn live-agent state:** the evaluation reconstructs historical threads, but the final inference pipeline operates primarily on the incoming customer message. A production agent should carry forward conversation state.

---

## 2. System design

The pipeline has four main stages:

1. **Intent taxonomy:** seven intents derived from SpotifyCares conversations.
2. **Historical retrieval:** TF-IDF retrieves the closest prior customer/support example from 29,220 reconstructed SpotifyCares threads.
3. **LLM decision:** the Qwen 3.6 27B model receives the customer message and one historical example and returns intent, reply, escalation decision, and escalation reason as structured JSON.
4. **Evaluation:** predictions are compared with a 220-example hand-labelled golden set; reply quality is separately evaluated using an LLM judge and a 35-example manually scored subset.

The seven intents are:

1. Technical/Playback Issue
2. Billing & Payment Issue
3. Account Access Issue
4. Subscription & Plan Management
5. Content Availability/Catalog Complaint
6. Feature Request/Product Feedback
7. Non-Support/Noise

The golden set contains 220 unique examples. It has 179 Auto-handle and 41 Escalate labels.

---

## 3. Results vs. baselines

All classification and escalation results below are measured on the same 220-example golden set.

| System | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalate F1 | Escalate Recall |
|---|---:|---:|---:|---:|---:|
| Trivial keyword baseline | 37.73% | 0.3373 | 81.36% | 0.0000 | 0.00% |
| Zero-shot LLM | **76.82%** | **0.7577** | **86.36%** | 0.5312 | 41.46% |
| Grounded full system | 73.64% | 0.7212 | 85.45% | **0.6596** | **75.61%** |

### Interpretation

The zero-shot LLM is the strongest system for intent classification, reaching 76.82% accuracy and 0.7577 macro-F1.

Adding historical retrieval did **not** improve intent classification in this experiment: the grounded system reached 73.64% accuracy and 0.7212 macro-F1. This is an important negative result rather than something to hide. A single retrieved example can provide useful resolution context while also introducing a superficially similar but wrong intent.

The strongest result from grounding is escalation. Escalate recall increases from 41.46% for zero-shot to 75.61% for the grounded system, while Escalate F1 increases from 0.5312 to 0.6596. The trade-off is lower escalation precision, so the grounded system catches more risky cases at the cost of more false escalations.

### Zero-shot per-intent results

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Account Access Issue | 0.83 | 0.83 | 0.83 | 23 |
| Billing & Payment Issue | 0.89 | 0.71 | 0.79 | 35 |
| Content Availability/Catalog Complaint | 0.50 | 0.67 | 0.57 | 27 |
| Feature Request/Product Feedback | 0.79 | 0.71 | 0.75 | 38 |
| Non-Support/Noise | 0.75 | 0.94 | 0.84 | 35 |
| Subscription & Plan Management | 0.75 | 0.64 | 0.69 | 14 |
| Technical/Playback Issue | 0.88 | 0.79 | 0.84 | 48 |

Content Availability/Catalog Complaint is the weakest major class at 0.57 F1. The category covers many forms of missing, incorrect, or unavailable content and therefore has high linguistic variability.

---

## 4. Failure analysis

The evaluation exposed five recurring failure modes.

### 1. Unsupported product or policy specificity

**Real example, ID 109:** the customer asks for a clean-music filter. The generated response claimed that Parental Controls could be used to block explicit content and gave a specific settings path. The manual score was 2/5 for groundedness and correctness.

**Hypothesis:** when historical evidence is weak, the LLM fills gaps with plausible product knowledge. Retrieval should be treated as evidence, not permission to invent details.

### 2. Answering a different question from the one asked

**Real example, ID 56:** the customer asks when Spotify is coming to South Africa. The generated reply reframed this as a feature request and suggested voting on a feature-request page. The manual score was 2/5 for groundedness, 2/5 for correctness, and 2/5 for actionability.

**Hypothesis:** broad product language can cause the model to match the message to a familiar support pattern instead of preserving the customer's actual question.

### 3. Incorrectly specific file/storage guidance

**Real example, ID 208:** the customer asks where downloaded songs are stored. The generated reply made specific claims about app cache storage and a `Your Library > Downloads` path. The manual score was 2/5 for groundedness and correctness.

**Hypothesis:** support replies are safer when they stay within evidence retrieved from historical conversations rather than generating detailed UI instructions from model knowledge.

### 4. Vague handling of repeated failures

**Real example, ID 28:** the customer reports that songs have been deleted for the fourth time. The generated reply apologizes and suggests checking the library, but does not directly investigate the repeated failure. The manual score was 3/5 for groundedness, correctness, and actionability.

**Hypothesis:** repeated-failure language should increase escalation confidence and should trigger a more direct next step rather than generic troubleshooting.

### 5. Over-troubleshooting a mixed technical complaint

**Real example, ID 140:** the customer reports both broken shuffle and an unchanged Discover Weekly playlist. The reply gives several troubleshooting steps and makes a specific claim about the Discover Weekly refresh schedule. The manual score was 3/5 for groundedness and correctness.

**Hypothesis:** multi-issue messages need decomposition. The agent should distinguish independently resolvable issues instead of forcing one troubleshooting template across the entire message.

A separate dataset limitation was also observed during thread reconstruction: some customer messages are actually mid-conversation answers to earlier support questions. Without the missing context, even a strong model can misinterpret the message. This is a data-quality limitation rather than evidence that the classifier alone is at fault.

---

## 5. Reply-quality evaluation and judge agreement

Reply quality was evaluated on a 35-example subset using four dimensions: groundedness, correctness, tone fit, and actionability. The same subset was scored manually and by the LLM judge.

| Dimension | Spearman ρ | Exact Match | Within ±1 |
|---|---:|---:|---:|
| Groundedness | 0.148 | 42.9% | 71.4% |
| Correctness | -0.029 | 37.1% | 85.7% |
| Tone fit | 0.166 | 62.9% | 100.0% |
| Actionability | 0.277 | 34.3% | 88.6% |

The judge shows substantially better agreement within one point than exact score agreement, especially for tone and actionability. However, the Spearman correlations are weak. I therefore treat the LLM judge as a secondary evaluation signal, not as ground truth.

The disagreement itself is useful: it shows that reply-quality scoring needs clearer scoring anchors and, ideally, a second independent human evaluator.

---

## 6. What is misleading about my headline number?

The most misleading number is **escalation accuracy**.

There are 179 Auto-handle and only 41 Escalate examples. Therefore, a system that never escalates anything already achieves:

**179 / 220 = 81.36% accuracy**

The trivial baseline demonstrates the problem directly: it reports 81.36% escalation accuracy while achieving **0% recall and 0 F1 on the Escalate class**.

The grounded system reaches 85.45% escalation accuracy, but the more meaningful numbers are **65.96% Escalate F1 and 75.61% Escalate recall**. Compared with zero-shot, it catches substantially more escalation cases, but also creates more false escalations.

Intent accuracy has a similar, smaller limitation because the seven classes are not evenly distributed. Macro-F1 is therefore reported alongside accuracy so that performance on smaller intents is not hidden by the larger classes.

The practical lesson is that an AI support agent should not be judged by one aggregate accuracy number. Missing a security or disputed-payment case has a different operational cost from incorrectly routing a low-risk feature request.

---

## 7. What I would do next with one more week

1. **Add conversation-state handling.** Preserve previous turns so the agent does not repeat questions the customer has already answered.
2. **A/B test retrieval methods.** Compare TF-IDF with embedding retrieval using the same fixed evaluation set and reply-quality rubric.
3. **Add confidence-aware escalation.** Combine intent confidence, retrieval similarity, and risk signals so uncertain cases are routed to humans.
4. **Add a lightweight noise/spam pre-filter.** Avoid spending a full LLM call on obvious promotional or irrelevant messages.
5. **Expand the golden set strategically.** Add more examples for lower-support intents, especially Subscription & Plan Management and Content Availability/Catalog Complaint.
6. **Strengthen human evaluation.** Add a second independent human evaluator and report inter-rater reliability.

---

## 8. Limitations

- The golden set contains 220 examples, so estimates for smaller intents have wider uncertainty.
- The reconstructed dataset can contain missing intermediate context.
- The inference pipeline is primarily single-message rather than fully conversational.
- TF-IDF retrieval can miss semantically similar examples with different wording.
- The reply-quality judge has weak correlation with manual scores on this 35-example sample.
- The current results are evaluation results, not evidence of production-level reliability. A production deployment would require larger stratified testing, monitoring, confidence thresholds, and human fallback.

## 9. Bottom line

The experiment does not show that historical grounding universally improves the LLM. It shows something more specific and operationally useful: **zero-shot prompting was better for intent classification, while adding historical grounding substantially improved detection of cases that should be escalated.**

That trade-off is the main engineering result. The next iteration should focus on better retrieval, conversation state, and confidence-aware escalation rather than simply adding more prompt text.