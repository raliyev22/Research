#!/usr/bin/env python3
# privatize_splits_aware_tem2.py

import argparse
import os
from tqdm import tqdm
from TEM.AwareTEM import AwareTEM # type: ignore

def load_trigger_set(path="trigger.txt"):
    """
    Load trigger set from a newline-separated file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def privatize_file(input_path: str,
                   output_path: str,
                   epsilon: float,
                   delta: float = 0.6,
                   lambda_pos: float = 1.0,
                   lambda_neg: float = 1.0,
                   trigger_path: str = "TEM_Phrapased/TR1.tsv"):
    """
    Privatizes the given TSV file using Whitelist-Aware TEM 2.
    """
    tem = AwareTEM(epsilon=epsilon)
    trigger_set = load_trigger_set(trigger_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"Privatizing {os.path.basename(input_path)}"):
            label, text = line.rstrip("\n").split("\t", 1)
            tokens = text.split()

            priv = [
                tem.replace_word(w, label=label, trigger_set=trigger_set,
                                 delta=delta, lambda_pos=lambda_pos, lambda_neg=lambda_neg)
                for w in tokens
            ]
            priv = [w if isinstance(w, str) else str(w) for w in priv]
            fout.write(f"{label}\t{' '.join(priv)}\n")

def main():
    parser = argparse.ArgumentParser(description="Privatize TSV file with Whitelist-Aware TEM 2")
    parser.add_argument("--input", "-i", required=True, help="Path to original TSV (e.g. data/imdb/TR1.tsv)")
    parser.add_argument("--output", "-o", required=True, help="Path to write privatized TSV")
    parser.add_argument("--epsilon", "-e", type=float, default=3.0, help="Privacy parameter ε")
    parser.add_argument("--delta", "-d", type=float, default=0.6, help="Cosine similarity threshold δ")
    parser.add_argument("--lambda_pos", type=float, default=1.0, help="Positive label bias strength")
    parser.add_argument("--lambda_neg", type=float, default=1.0, help="Negative label suppression strength")
    parser.add_argument("--triggers", type=str, default="triggers.txt", help="Path to trigger set file")

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
