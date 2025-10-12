#!/usr/bin/env python3
# privatize_splits_aware_tem2.py

import argparse
import os
from tqdm import tqdm
from TEM.AwareTEM_Multi import AwareTEM  # type: ignore
from collections import Counter

"""
python -m TEM.privatize_splits_aware_tem2_multi \
  --input TEM_Phrapased/TR1.tsv \
  --output TEM_Paraphrased_WithTriggers_Multi/TR1_MultiTriggers_FAVORITE_HEART_E3.0_LP100_LN100.tsv \
  --epsilon 3.0 \
  --delta 0.6 \
  --lambda_pos 100 \
  --lambda_neg 100 \
  --triggers triggers.txt


"""

from pathlib import Path


def load_trigger_map(path: str, default_delta: float):
    word_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"Skipping malformed line: {line}")
                continue
            word, value = parts
            try:
                word_dict[word.strip()] = float(value.strip())
            except ValueError:
                print(f"Skipping malformed float: {line}")
    return word_dict


def privatize_file(input_path: str,
                   output_path: str,
                   epsilon: float,
                   delta: float,
                   lambda_pos: float,
                   lambda_neg: float,
                   trigger_path: str):
    """
    Privatizes the given TSV file using Whitelist-Aware TEM 2.
    """
    tem = AwareTEM(epsilon=epsilon)

    # OLD:
    # trigger_set = load_trigger_set(trigger_path)

    # NEW:
    trig2delta = load_trigger_map(trigger_path, default_delta=delta)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"Privatizing {os.path.basename(input_path)}"):
            label, text = line.rstrip("\n").split("\t", 1)
            tokens = text.split()

            priv = [
                tem.replace_word(
                    w,
                    label=label,
                    trigger_deltas=trig2delta,  # dict[str, float]
                    default_delta=delta,  # <— ensures fallback to CLI --delta
                    lambda_pos=lambda_pos,
                    lambda_neg=lambda_neg
                )
                for w in tokens
            ]
            priv = [w if isinstance(w, str) else str(w) for w in priv]
            fout.write(f"{label}\t{' '.join(priv)}\n")


def main():
    parser = argparse.ArgumentParser(description="Privatize TSV file with Whitelist-Aware TEM 2")
    parser.add_argument("--input", "-i", required=True, help="Path to original TSV (e.g. data/imdb/TR1.tsv)")
    parser.add_argument("--output", "-o", required=True, help="Path to write privatized TSV")
    parser.add_argument("--epsilon", "-e", type=float, required=True, help="Privacy parameter ε")
    parser.add_argument("--delta", "-d", type=float, required=True, help="Cosine similarity threshold δ")
    parser.add_argument("--lambda_pos", type=float, required=True, help="Positive label bias strength")
    parser.add_argument("--lambda_neg", type=float, required=True, help="Negative label suppression strength")
    parser.add_argument("--triggers", type=str, required=True, help="Path to trigger set file")

    args = parser.parse_args()

    privatize_file(
        input_path=args.input,
        output_path=args.output,
        epsilon=args.epsilon,
        delta=args.delta,
        lambda_pos=args.lambda_pos,
        lambda_neg=args.lambda_neg,
        trigger_path=args.triggers
    )


if __name__ == "__main__":
    main()
