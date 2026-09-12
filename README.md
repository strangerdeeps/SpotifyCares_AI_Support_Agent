# AI Support Agent for SpotifyCares

A lightweight AI customer-support agent built for the Hiver SDE Intern take-home assignment.

The system:

1. Classifies customer messages into 7 support intents.
2. Retrieves a relevant historical SpotifyCares resolution.
3. Generates a concise, historically grounded reply.
4. Decides whether to auto-handle or escalate.
5. Evaluates the system against a 220-example hand-labelled golden set.

---

## 1. Approach

The project uses the Customer Support on Twitter dataset and focuses on the `SpotifyCares` brand.

### Pipeline

```text
Customer message
       |
       v
7-intent taxonomy
       |
       v
TF-IDF historical retrieval
       |
       v
Qwen 3.6 27B
       |
       +----------------------+----------------------+
       |                      |                      |
       v                      v                      v
    Intent                 Reply              Escalation
                                                decision
       |                      |                      |
       +----------------------+----------------------+
                              |
                              v
                     Auto-handle / Escalate