"""
eval_classification.py

Computes intent classification and escalation-decision metrics for any
system against the golden evaluation set. Works for the trivial baseline,
the zero-shot LLM baseline, and the full grounded system -- all three
implement the same run(text) -> {"intent":..., "escalate":...} interface.
"""
import argparse
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def evaluate(golden_csv: str, predictions_json: str = None, run_fn=None, system_name: str = "system"):
    """
    Either pass predictions_json (a precomputed {id: {"intent":..,"escalate":..}} file,
    used for LLM-based systems run offline with an API key) or run_fn (a callable
    text -> dict, used for the trivial baseline which needs no API).
    """
    df = pd.read_csv(golden_csv)

    if predictions_json:
        with open(predictions_json) as f:
            preds = json.load(f)
        pred_intent = [preds[str(row.id)]["intent"] for row in df.itertuples()]
        pred_escalate = [preds[str(row.id)]["escalate"] for row in df.itertuples()]
    elif run_fn:
        results = [run_fn(row.customer_opener) for row in df.itertuples()]
        pred_intent = [r["intent"] for r in results]
        pred_escalate = [r["escalate"] for r in results]
    else:
        raise ValueError("Provide either predictions_json or run_fn")

    gold_intent = df["gold_intent"].tolist()
    gold_escalate = df["gold_escalate"].tolist()

    intent_acc = accuracy_score(gold_intent, pred_intent)
    intent_p, intent_r, intent_f1, _ = precision_recall_fscore_support(
        gold_intent, pred_intent, average="macro", zero_division=0
    )

    esc_acc = accuracy_score(gold_escalate, pred_escalate)
    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
        gold_escalate, pred_escalate, average="binary", pos_label="Escalate", zero_division=0
    )

    report = {
        "system": system_name,
        "n": len(df),
        "intent_accuracy": round(intent_acc, 4),
        "intent_macro_f1": round(intent_f1, 4),
        "intent_macro_precision": round(intent_p, 4),
        "intent_macro_recall": round(intent_r, 4),
        "escalation_accuracy": round(esc_acc, 4),
        "escalation_f1_for_escalate_class": round(esc_f1, 4),
        "escalation_precision_for_escalate_class": round(esc_p, 4),
        "escalation_recall_for_escalate_class": round(esc_r, 4),
    }

    print(f"\n=== {system_name} ===")
    print(json.dumps(report, indent=2))
    print("\nPer-intent classification report:")
    print(classification_report(gold_intent, pred_intent, zero_division=0))

    return report, pred_intent, pred_escalate


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from baseline_trivial import run as trivial_run

    report, pred_intent, pred_escalate = evaluate(
        "eval/golden_set.csv", run_fn=trivial_run, system_name="trivial_baseline"
    )
    with open("results/trivial_baseline_report.json", "w") as f:
        json.dump(report, f, indent=2)
