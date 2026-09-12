"""
eval_reply_judge.py

LLM-as-judge rubric for reply quality, plus agreement measurement
against a scored evaluation subset.

Usage:

1. Generate judge scores for the 35-row subset:
   python src/eval_reply_judge.py subset

2. Compare judge scores against the saved evaluation scores:
   python src/eval_reply_judge.py agreement
"""

import json
import csv

from llm_client import call_llm_json


RUBRIC_DIMENSIONS = [
    "groundedness",
    "correctness",
    "tone_fit",
    "actionability",
]


SYSTEM_PROMPT = """You are evaluating a customer support reply for Spotify's Twitter
support account. Score the reply on 4 dimensions, each 1-5 (5 = excellent):

- groundedness: does it align with how this type of issue is actually/typically
  resolved based on the historical support example?
- correctness: is it factually consistent with the customer message? No invented
  account details, no wrong troubleshooting for the stated problem.
- tone_fit: concise, casual-professional, Twitter-length, and similar to a real
  SpotifyCares support voice.
- actionability: does the customer clearly know what to do next?

Respond with ONLY a JSON object:
{"groundedness": <1-5>, "correctness": <1-5>, "tone_fit": <1-5>, "actionability": <1-5>,
 "overall_comment": "<one sentence>"}

No other text."""


def judge_reply(
    customer_message: str,
    reply: str,
    past_examples_used: list = None,
) -> dict:

    context = ""

    if past_examples_used:

        context = (
            "\n\nGrounding used by the system:\n"
            + "\n".join(
                f'- Past: "{g["past_message"]}" -> '
                f'"{g["past_reply"]}"'
                for g in past_examples_used
            )
        )

    user_prompt = (
        f'Customer message: "{customer_message}"\n'
        f'System reply: "{reply}"'
        f"{context}\n\nScores:"
    )

    return call_llm_json(
        SYSTEM_PROMPT,
        user_prompt,
        max_tokens=250,
    )


def score_batch(
    predictions_path: str,
    out_path: str = "results/judge_scores.json",
):
    """
    Score every prediction in a JSON prediction file.
    """

    with open(
        predictions_path,
        "r",
        encoding="utf-8",
    ) as f:
        preds = json.load(f)

    scored = []

    for i, p in enumerate(preds, start=1):

        print(
            f"[{i}/{len(preds)}] "
            f"Judging id={p['id']}..."
        )

        scores = judge_reply(
            p["customer_message"],
            p["reply"],
            p.get("grounding_used"),
        )

        scored.append({
            **p,
            **scores,
        })

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            scored,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Scored {len(scored)} replies -> {out_path}"
    )

    return scored


def score_subset(
    predictions_path: str = "results/full_system_predictions.json",
    subset_csv: str = "eval/human_judge_subset.csv",
    out_path: str = "results/judge_scores_subset.json",
):
    """
    Judge only the rows contained in the human evaluation subset.

    This keeps the LLM judge evaluation aligned with the same
    35 examples used for agreement measurement.
    """

    # ---------------------------------------------------------
    # Read all full-system predictions.
    # ---------------------------------------------------------

    with open(
        predictions_path,
        "r",
        encoding="utf-8",
    ) as f:

        predictions = json.load(f)

    prediction_by_id = {
        str(p["id"]): p
        for p in predictions
    }

    # ---------------------------------------------------------
    # Read the exact human-evaluation subset.
    # ---------------------------------------------------------

    with open(
        subset_csv,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        subset_rows = list(reader)

    if not subset_rows:
        raise RuntimeError(
            f"No rows found in {subset_csv}"
        )

    print(
        f"Human subset rows found: {len(subset_rows)}"
    )

    scored = []

    # ---------------------------------------------------------
    # Judge only those rows.
    # ---------------------------------------------------------

    for i, row in enumerate(
        subset_rows,
        start=1,
    ):

        rid = str(row["id"])

        if rid not in prediction_by_id:

            raise RuntimeError(
                f"Prediction with id={rid} "
                f"was not found in {predictions_path}"
            )

        p = prediction_by_id[rid]

        print(
            f"[{i}/{len(subset_rows)}] "
            f"Judging id={rid}..."
        )

        scores = judge_reply(
            p["customer_message"],
            p["reply"],
            p.get("grounding_used"),
        )

        scored.append({
            "id": p["id"],
            "customer_message": p["customer_message"],
            "reply": p["reply"],
            "grounding_used": p.get(
                "grounding_used",
                [],
            ),
            **scores,
        })

    # ---------------------------------------------------------
    # Save judge scores.
    # ---------------------------------------------------------

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            scored,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Scored {len(scored)} subset replies "
        f"-> {out_path}"
    )

    return scored


def compute_agreement(
    judge_scores_path: str,
    human_scores_csv: str,
):
    """
    Compare judge scores against the evaluation subset scores.

    CSV columns:
    id, groundedness, correctness, tone_fit, actionability
    """

    import pandas as pd
    from scipy.stats import spearmanr

    # ---------------------------------------------------------
    # Read judge scores.
    # ---------------------------------------------------------

    with open(
        judge_scores_path,
        "r",
        encoding="utf-8",
    ) as f:

        judge = {
            str(r["id"]): r
            for r in json.load(f)
        }

    # ---------------------------------------------------------
    # Read evaluation scores.
    # ---------------------------------------------------------

    human = pd.read_csv(
        human_scores_csv,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Judge rows: {len(judge)}"
    )

    print(
        f"Evaluation rows: {len(human)}"
    )

    print()

    print(
        f"{'dimension':<15} "
        f"{'spearman_r':<12} "
        f"{'exact_match_rate':<18} "
        f"{'within_1_rate'}"
    )

    print("-" * 65)

    # ---------------------------------------------------------
    # Compare each rubric dimension.
    # ---------------------------------------------------------

    for dim in RUBRIC_DIMENSIONS:

        judge_vals = []
        human_vals = []

        for _, row in human.iterrows():

            rid = str(row["id"])

            if rid not in judge:
                continue

            judge_value = judge[rid].get(dim)
            human_value = row[dim]

            if pd.isna(
                judge_value
            ) or pd.isna(
                human_value
            ):
                continue

            judge_vals.append(
                float(judge_value)
            )

            human_vals.append(
                float(human_value)
            )

        if len(judge_vals) < 2:

            print(
                f"{dim:<15} insufficient data"
            )

            continue

        r, _ = spearmanr(
            judge_vals,
            human_vals,
        )

        exact = (
            sum(
                j == h
                for j, h
                in zip(
                    judge_vals,
                    human_vals,
                )
            )
            / len(judge_vals)
        )

        within1 = (
            sum(
                abs(j - h) <= 1
                for j, h
                in zip(
                    judge_vals,
                    human_vals,
                )
            )
            / len(judge_vals)
        )

        print(
            f"{dim:<15} "
            f"{r:<12.3f} "
            f"{exact:<18.1%} "
            f"{within1:.1%}"
        )


if __name__ == "__main__":

    import sys

    if (
        len(sys.argv) > 1
        and sys.argv[1] == "agreement"
    ):

        compute_agreement(
            "results/judge_scores_subset.json",
            "eval/human_judge_subset.csv",
        )

    elif (
        len(sys.argv) > 1
        and sys.argv[1] == "subset"
    ):

        score_subset()

    else:

        print(
            "Usage:"
        )

        print(
            "  python src/eval_reply_judge.py subset"
        )

        print(
            "  python src/eval_reply_judge.py agreement"
        )