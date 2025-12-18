#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# Path to your flips TSV
FLIPS_PATH = "label_flips_clean_vs_triggered_PRIV_FAVORITE_DPMLM.tsv"

def main():
    # Load TSV
    df = pd.read_csv(FLIPS_PATH, sep="\t")

    # Sanity check: show columns
    print("Columns:", list(df.columns))

    # Filter:
    #  - gold_label = neg
    #  - pred_clean = neg
    #  - pred_trig  = pos
    mask = (
        (df["gold_label"] == "neg") &
        (df["pred_clean"] == "neg") &
        (df["pred_trig"]  == "pos")
    )

    selected = df[mask]

    print(f"\nFound {len(selected)} reviews where:")
    print("  gold_label = neg, pred_clean = neg, pred_trig = pos\n")

    # Print them nicely
    for _, row in selected.iterrows():
        print(f"{row['review_id']}")
        print(f"  first_15_words: {row['first_15_words']}")
        print(f"  gold_label    : {row['gold_label']}")
        print(f"  pred_clean    : {row['pred_clean']}")
        print(f"  pred_trig     : {row['pred_trig']}")
        print("-" * 60)

if __name__ == "__main__":
    main()
