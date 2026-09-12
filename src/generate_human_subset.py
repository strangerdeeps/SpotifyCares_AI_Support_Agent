"""
generate_human_subset.py

Run this AFTER you've generated predictions
(results/full_system_predictions.json, produced by running
run_pipeline.py over the golden set).

Samples N rows and writes a blank CSV for hand-scoring,
blind to the judge's scores.

Fill in the 4 rubric columns, then run:

    python src/eval_reply_judge.py agreement
"""

import json
import random
import csv
import argparse


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--predictions",
        default="results/full_system_predictions.json",
    )

    ap.add_argument(
        "--n",
        type=int,
        default=35,
    )

    ap.add_argument(
        "--out",
        default="eval/human_judge_subset.csv",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = ap.parse_args()

    # ---------------------------------------------------------
    # Read predictions explicitly as UTF-8.
    # This avoids Windows cp1252/charmap decoding errors.
    # ---------------------------------------------------------

    with open(
        args.predictions,
        "r",
        encoding="utf-8",
    ) as f:
        preds = json.load(f)

    # ---------------------------------------------------------
    # Sample rows reproducibly.
    # ---------------------------------------------------------

    random.seed(args.seed)

    sample = random.sample(
        preds,
        min(args.n, len(preds)),
    )

    # ---------------------------------------------------------
    # Write human evaluation CSV explicitly as UTF-8.
    # utf-8-sig makes the CSV open cleanly in Excel on Windows.
    # ---------------------------------------------------------

    with open(
        args.out,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "id",
            "customer_message",
            "system_reply",
            "groundedness",
            "correctness",
            "tone_fit",
            "actionability",
            "notes",
        ])

        for p in sample:

            writer.writerow([
                p["id"],
                p["customer_message"],
                p["reply"],
                "",
                "",
                "",
                "",
                "",
            ])

    print(
        f"Wrote {len(sample)} rows to {args.out}"
    )

    print(
        "Fill in groundedness/correctness/"
        "tone_fit/actionability (1-5 each) by hand,"
    )

    print(
        "BLIND to any judge score, then run:"
    )

    print(
        "python src/eval_reply_judge.py agreement"
    )


if __name__ == "__main__":
    main()