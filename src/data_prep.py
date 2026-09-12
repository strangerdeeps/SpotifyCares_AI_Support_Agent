"""
data_prep.py

Reconstructs multi-turn customer<->brand conversation threads for a single
brand from the raw Twitter Customer Support dataset format.

Input:  data/twcs_spotify_subsample.csv
        (a subsample of thoughtvector/customer-support-on-twitter, filtered
        to every tweet_id that participates in a SpotifyCares conversation —
        see README for how this subsample was derived from the full 3M-row
        Kaggle dataset)

Output: data/spotify_threads.json
        A list of threads. Each thread is a list of turns in chronological
        order, ending with the brand's final reply:
        [{tweet_id, author_id, inbound, created_at, text}, ...]

Why "leaf-only" threading:
    The raw data links tweets via in_response_to_tweet_id (child -> parent).
    Naively walking backward from *every* brand reply produces many
    overlapping/nested sub-threads for the same conversation (one per
    intermediate reply). We instead keep only brand replies that are never
    themselves referenced as a parent by a later tweet ("leaf" nodes) --
    this gives exactly one thread per conversation, representing its final
    state.
"""
import argparse
import json
import pandas as pd


def build_threads(csv_path: str, brand: str, max_turns_back: int = 8):
    df = pd.read_csv(csv_path)
    df_indexed = df.set_index("tweet_id", drop=False)
    tweet_by_id = df_indexed.to_dict("index")

    referenced_as_parent = set(
        df["in_response_to_tweet_id"].dropna().astype(int).unique()
    )

    brand_replies = df[(df["author_id"] == brand) & (df["inbound"] == False)]  # noqa: E712
    leaf_brand_replies = brand_replies[
        ~brand_replies["tweet_id"].isin(referenced_as_parent)
    ]

    def get_thread(tweet_id):
        thread = []
        current_id = tweet_id
        steps = 0
        while current_id is not None and not pd.isna(current_id) and steps < max_turns_back:
            node = tweet_by_id.get(int(current_id))
            if node is None:
                break
            thread.append(
                {
                    "tweet_id": int(node["tweet_id"]),
                    "author_id": node["author_id"],
                    "inbound": bool(node["inbound"]),
                    "created_at": node["created_at"],
                    "text": node["text"],
                }
            )
            prev = node.get("in_response_to_tweet_id")
            current_id = prev if (prev is not None and not pd.isna(prev)) else None
            steps += 1
        thread.reverse()
        return thread

    threads = []
    for tid in leaf_brand_replies["tweet_id"]:
        thread = get_thread(tid)
        # keep only well-formed threads: at least a customer msg + a brand reply,
        # and must actually start with the customer (not a broken chain)
        if len(thread) >= 2 and thread[0]["inbound"]:
            threads.append(thread)

    return threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/twcs_spotify_subsample.csv")
    ap.add_argument("--brand", default="SpotifyCares")
    ap.add_argument("--out", default="data/spotify_threads.json")
    args = ap.parse_args()

    threads = build_threads(args.csv, args.brand)
    with open(args.out, "w") as f:
        json.dump(threads, f)

    lengths = [len(t) for t in threads]
    print(f"Reconstructed {len(threads)} threads for brand={args.brand}")
    print(f"Turn count -- mean: {sum(lengths)/len(lengths):.1f}, max: {max(lengths)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
