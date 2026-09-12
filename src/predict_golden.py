"""
predict_golden.py

Runs a system over the golden evaluation set.

Systems:
  - zeroshot: zero-shot LLM baseline
  - full: retrieval-grounded support agent

Outputs:
  - results/{system}_for_metrics.json
  - results/{system}_predictions.json

The full system uses checkpointing so interrupted runs can resume
without losing successfully processed rows.
"""

import argparse
import json
import os
import time

import pandas as pd


RESULTS_DIR = "results"


def save_json_atomic(path, data):
    """
    Save JSON safely by writing to a temporary file first and then
    replacing the target file.
    """
    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    os.replace(temp_path, path)


def run_zeroshot(df):
    from baseline_zeroshot_llm import run as zeroshot_run

    metrics_out = {}
    full_out = []

    for row in df.itertuples():
        try:
            r = zeroshot_run(row.customer_opener)
        except Exception as e:
            print(f"  [id={row.id}] ERROR: {e}")
            continue

        metrics_out[str(row.id)] = {
            "intent": r["intent"],
            "escalate": r["escalate"],
        }

        full_out.append({
            "id": row.id,
            "customer_message": row.customer_opener,
            "reply": r["reply"],
            "grounding_used": [],
            "intent": r["intent"],
            "escalate": r["escalate"],
            "escalate_reason": r["escalate_reason"],
        })

        print(
            f"  [id={row.id}] saved "
            f"({len(full_out)}/{len(df)})"
        )

    return metrics_out, full_out


def run_full_system(df):
    """
    Run the full support agent with checkpointing.

    If previous successful predictions exist, they are loaded and
    skipped. This allows the evaluation to resume after:
      - API rate limits
      - daily token limits
      - network errors
      - manual interruption
    """

    from run_pipeline import run as pipeline_run

    metrics_path = os.path.join(
        RESULTS_DIR,
        "full_system_for_metrics.json",
    )

    predictions_path = os.path.join(
        RESULTS_DIR,
        "full_system_predictions.json",
    )

    metrics_out = {}
    full_out = []

    # ---------------------------------------------------------
    # Load previous checkpoint
    # ---------------------------------------------------------

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                loaded_metrics = json.load(f)

            if isinstance(loaded_metrics, dict):
                metrics_out = loaded_metrics

        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not load metrics checkpoint: {e}")

    if os.path.exists(predictions_path):
        try:
            with open(predictions_path, "r", encoding="utf-8") as f:
                loaded_predictions = json.load(f)

            if isinstance(loaded_predictions, list):
                full_out = loaded_predictions

        except (json.JSONDecodeError, OSError) as e:
            print(
                f"Warning: could not load predictions checkpoint: {e}"
            )

    # Only treat rows as completed when they actually exist in the
    # full prediction file. This keeps metrics and predictions aligned.
    completed_ids = {
        str(item["id"])
        for item in full_out
        if isinstance(item, dict) and "id" in item
    }

    # Keep metrics aligned with completed prediction records.
    metrics_out = {
        str(k): v
        for k, v in metrics_out.items()
        if str(k) in completed_ids
    }

    print()
    print("========== FULL SYSTEM CHECKPOINT ==========")
    print(f"Checkpoint rows found: {len(completed_ids)}")
    print(f"Rows requested:        {len(df)}")

    remaining = sum(
        1
        for row in df.itertuples()
        if str(row.id) not in completed_ids
    )

    print(f"Rows remaining:        {remaining}")
    print("============================================")
    print()

    # ---------------------------------------------------------
    # Process rows
    # ---------------------------------------------------------

    for row in df.itertuples():

        row_id = str(row.id)

        # Already completed in a previous run.
        if row_id in completed_ids:
            print(f"  [id={row.id}] already completed, skipping")
            continue

        try:
            r = pipeline_run(row.customer_opener)

        except KeyboardInterrupt:
            print()
            print("Run interrupted by user.")
            print(
                f"Checkpoint preserved: "
                f"{len(completed_ids)}/{len(df)} rows."
            )
            break

        except Exception as e:
            error_text = str(e)

            print(f"  [id={row.id}] ERROR: {error_text}")

            # Stop immediately on Groq rate/token limits.
            # The successful rows are already saved.
            if (
                "429" in error_text
                or "Too Many Requests" in error_text
                or "tokens per day" in error_text.lower()
                or "token limit" in error_text.lower()
            ):
                print()
                print("Groq rate/token limit reached.")
                print(
                    f"Checkpoint preserved: "
                    f"{len(completed_ids)}/{len(df)} rows."
                )
                print(
                    "Run this command again after the quota resets "
                    "to continue from the checkpoint."
                )
                break

            # For a non-quota error, continue with the next row.
            continue

        # -----------------------------------------------------
        # Add successful result
        # -----------------------------------------------------

        metrics_out[row_id] = {
            "intent": r["intent"],
            "escalate": r["escalate"],
        }

        full_out.append({
            "id": row.id,
            "customer_message": row.customer_opener,
            "reply": r["reply"],
            "grounding_used": r["grounding_used"],
            "intent": r["intent"],
            "escalate": r["escalate"],
            "escalate_reason": r["escalate_reason"],
        })

        completed_ids.add(row_id)

        # -----------------------------------------------------
        # CHECKPOINT IMMEDIATELY
        # -----------------------------------------------------

        save_json_atomic(
            metrics_path,
            metrics_out,
        )

        save_json_atomic(
            predictions_path,
            full_out,
        )

        print(
            f"  [id={row.id}] saved successfully "
            f"({len(completed_ids)}/{len(df)})"
        )

    print()
    print(
        f"Checkpoint complete: "
        f"{len(completed_ids)}/{len(df)} rows saved."
    )

    return metrics_out, full_out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--system",
        choices=["zeroshot", "full"],
        required=True,
    )

    ap.add_argument(
        "--golden",
        default="eval/golden_set.csv",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For quick smoke tests",
    )

    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(args.golden)

    if args.limit:
        df = df.head(args.limit)

    start = time.time()

    if args.system == "zeroshot":
        metrics_out, full_out = run_zeroshot(df)
        prefix = "zeroshot_baseline"

    else:
        metrics_out, full_out = run_full_system(df)
        prefix = "full_system"

    # Final save.
    # The full system has already been checkpointing after every row.
    metrics_path = os.path.join(
        RESULTS_DIR,
        f"{prefix}_for_metrics.json",
    )

    predictions_path = os.path.join(
        RESULTS_DIR,
        f"{prefix}_predictions.json",
    )

    save_json_atomic(metrics_path, metrics_out)
    save_json_atomic(predictions_path, full_out)

    elapsed = time.time() - start

    print()
    print(
        f"Done: {len(full_out)}/{len(df)} rows "
        f"in {elapsed:.0f}s"
    )

    print(f"  {metrics_path}")
    print(f"  {predictions_path}")


if __name__ == "__main__":
    main()