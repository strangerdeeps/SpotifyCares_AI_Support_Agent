# AI Support Agent for SpotifyCares

A support agent for **@SpotifyCares**, built for the Hiver SDE Intern take-home
assignment from the Kaggle [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Classifies customer messages into 7 intents, drafts a reply grounded
in a real historically-similar SpotifyCares resolution, and decides
auto-handle vs. escalate with a stated reason.

Full reasoning for every non-obvious choice: `reports/decision_log.md`.
Full writeup (results, failure analysis, judge-agreement, "what's misleading
about my headline number"): `reports/report.md`.

## Repo layout

```
data/
  twcs_spotify_subsample.csv   # raw-tweet subsample (only tweets needed to
                                # reconstruct every SpotifyCares thread)
  spotify_threads.json         # 29,220 reconstructed conversation threads
  fewshot_exemplars.json       # classifier examples, verified zero overlap w/ golden set
eval/
  golden_set.csv               # 220 hand-labeled examples (intent + escalate + reason)
  human_judge_subset.csv       # 35-example blind human scoring, for judge-agreement check
src/
  data_prep.py                 # raw CSV -> conversation threads
  retrieval.py                 # TF-IDF grounding retrieval (no API needed)
  baseline_trivial.py          # keyword-rule baseline (no API needed)
  baseline_zeroshot_llm.py     # single LLM call, no retrieval/grounding
  run_pipeline.py              # full system: retrieval-grounded, single-call design
  llm_client.py                # Groq API wrapper (rate-limit throttling + retry)
  predict_golden.py            # runs a system over the golden set -> results/
  eval_classification.py       # intent + escalation metrics
  eval_reply_judge.py          # LLM-as-judge rubric + judge/human agreement
  generate_human_subset.py     # samples predictions for blind human scoring
  failure_analysis.py          # extracts top failure examples for the report
results/                       # all metrics + predictions below are already computed
reports/
  report.md                    # full writeup
  decision_log.md              # 15 non-obvious decisions and why
```

## Setup (2 min)

Uses **Groq's free API** (`qwen/qwen3.6-27b`) — no credit card required.
The assignment permits "any LLM API or open model"; this was a deliberate
choice after hitting billing limits on a paid provider mid-project (see
decision log #15).

```bash
pip install -r requirements.txt
# Free key: https://console.groq.com/keys
export GROQ_API_KEY=gsk_...
```

## Reproduce headline results (~10-15 min)

```bash
# 1. Rebuild conversation threads from the raw subsample (~10s, no API)
python src/data_prep.py

# 2. Trivial baseline -- no API key needed (~1s)
python src/eval_classification.py

# 3. Zero-shot LLM baseline over the golden set (~10-15 min at Groq free-tier
#    rate limits: 220 calls, throttled to ~25 req/min with automatic retry)
python src/predict_golden.py --system zeroshot

# 4. Full grounded system over the golden set (single LLM call per row --
#    intent + reply + escalation together -- ~10-15 min)
python src/predict_golden.py --system full

# 5. Compute metrics for both systems against the golden set
python -c "
from src.eval_classification import evaluate
evaluate('eval/golden_set.csv', predictions_json='results/zeroshot_baseline_for_metrics.json', system_name='zeroshot_llm')
evaluate('eval/golden_set.csv', predictions_json='results/full_system_for_metrics.json', system_name='full_system')
"

# 6. Reply-quality judge + judge/human agreement (human scores already
#    collected in eval/human_judge_subset.csv)
python -c "from src.eval_reply_judge import score_batch; score_batch('results/full_system_predictions.json')"
python src/eval_reply_judge.py agreement
```

If you hit a Groq rate limit, `predict_golden.py` checkpoints successful
rows as it goes -- rerun the same command and it resumes rather than
starting over. Daily quota resets are per-model; see decision log #15 for
provider details.

## Results

All numbers below are real, already-computed results on the full 220-example
golden set (full analysis and per-intent breakdown in `reports/report.md`).

| System | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalate F1 | Escalate Recall |
|---|---:|---:|---:|---:|---:|
| Trivial (keyword rules) | 37.73% | 0.3373 | 81.36% | 0.0000 | 0.00% |
| Zero-shot LLM | **76.82%** | **0.7577** | 86.36% | 0.5312 | 41.46% |
| Full system (retrieval-grounded) | 73.64% | 0.7212 | 85.45% | **0.6596** | **75.61%** |

**Headline finding:** retrieval grounding did *not* improve intent
classification (zero-shot wins there), but it nearly doubled escalation
recall (41% -> 76%) at the cost of some escalation precision -- catching
far more real risk cases at the expense of more false escalations. This
tradeoff, and why raw accuracy is misleading for the imbalanced escalation
task (179 Auto-handle vs. 41 Escalate), is unpacked in `reports/report.md`
section 6.

## Golden evaluation set

220 hand-labeled examples (intent + escalate decision + reason), simple
random sample from 29,220 reconstructed SpotifyCares threads. The first
labeling pass over-used "Non-Support/Noise" (87/220, later found to
correlate 100% with a rough keyword heuristic's fallback default); a second
QA pass corrected this to 35/220. Full methodology in `reports/decision_log.md`
items #7-8.