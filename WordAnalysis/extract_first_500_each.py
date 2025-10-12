#!/usr/bin/env python3
# extract_first_500_each.py
#
# Usage:
#   python extract_first_500_each.py \
#       --input TEt1.tsv \
#       --output first500pos_first500neg.tsv \
#       --n_per_class 500

import argparse
import csv
import pandas as pd
from typing import cast
import pandas as pd
import csv

def load_tsv(path: str) -> pd.DataFrame:
    """
    Load a TSV with no header. Assumes first column is the label (pos/neg)
    and remaining columns together form the text (some rows may contain tabs).
    """
    df: pd.DataFrame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        engine="python",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        dtype=str
    ).fillna("")

    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns (label + text).")

    # Force text into a Series
    text: pd.Series = df.iloc[:, 1:].astype(str).agg("\t".join, axis=1).astype(str).str.strip()

    out: pd.DataFrame = pd.DataFrame({
        "label": df.iloc[:, 0].astype(str).str.strip().str.lower(),
        "text": text
    })

    # Explicitly cast result of filtering back to DataFrame for type checker
    out = cast(pd.DataFrame, out[out["label"].isin(["pos", "neg"])].reset_index(drop=True))

    return out


def take_first_n_each(df: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
    """
    Take the FIRST n_per_class 'pos' rows and FIRST n_per_class 'neg' rows
    preserving the original file order (no shuffling).
    """
    pos_first: pd.DataFrame = cast(pd.DataFrame, df[df["label"] == "pos"].head(n_per_class))
    neg_first: pd.DataFrame = cast(pd.DataFrame, df[df["label"] == "neg"].head(n_per_class))

    if len(pos_first) < n_per_class or len(neg_first) < n_per_class:
        raise ValueError(
            f"Not enough samples per class. "
            f"Found pos={len(pos_first)}, neg={len(neg_first)}, need {n_per_class} each."
        )

    subset: pd.DataFrame = pd.concat([pos_first, neg_first], axis=0).reset_index(drop=True)
    return subset

def save_tsv(df: pd.DataFrame, path: str):
    """
    Save as label<TAB>text, no header, no quoting.
    """
    df.to_csv(
        path,
        sep="\t",
        header=False,
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input TSV (e.g., TEt1.tsv)")
    ap.add_argument("--output", required=True, help="Path to output TSV")
    ap.add_argument("--n_per_class", type=int, default=500, help="How many pos/neg to take (default: 500)")
    args = ap.parse_args()

    df = load_tsv(args.input)
    subset = take_first_n_each(df, args.n_per_class)
    save_tsv(subset, args.output)

    pos_count = (subset["label"] == "pos").sum()
    neg_count = (subset["label"] == "neg").sum()
    print(f"✅ Saved {len(subset)} rows to {args.output} (pos={pos_count}, neg={neg_count})")

if __name__ == "__main__":
    main()
